# Handoff: rebuild the anchor instead of substituting a real frame

This brief is for a fresh session. Read `README.md` first (it describes how the tool works), then this document. `docs/HANDOFF.md` is the original build brief and gives background on the design and on what has been validated.

## Where things stand

`frame_dedup` is a working end-to-end tool: `python -m frame_dedup in.mkv -o out.mkv --verify --report r.json`. It finds the inserted duplicate frames and removes them by H.264 bitstream surgery, with no re-encode. Duplicates are removed by dropping pictures, renumbering `frame_num`, and repairing the few macroblocks whose prediction changed with I_PCM patches. Everything else in the output decodes bit-identical to the original.

The weak spot is **drop rule 6 in `frame_dedup/plan.py` (`choose_drops`)**:

- **The case:** a duplicate pair whose later twin is an IDR. The tool never drops an IDR.
- **Rule 3 often avoids it:** it drops the earlier twin when nothing references it (header-only check).
- **When rule 3 fails:** the tool drops a *different* real frame near the pair, a "visible substitute".

On `smoke.mkv` that happens **300 times**. All 300 are the same shape, P then IDR (`types == 'PIi'` in `work/smoke_report.json` → `substitute_drops`). The first one is pair display 720, with substitute display 717.

**The owner's verdict:** a substitute is the worst outcome. Each one creates *two* artefacts, a skipped frame and a frame held too long, so it doubles the judder we are trying to remove. Any solution that drops the actual duplicate is preferable. Small, compression-noise-level differences confined to a few non-reference B-frames are explicitly acceptable.

## The case in detail

Display order, with K = the earlier twin (a P) and D = the later twin (an IDR):

```
display:  P0  B1 B2 B3  K | D(IDR)  b  b  b  P ...
decode:   P0 ... K  B1 B2 B3  D  P  b b b ...
```

- **D** starts a new IDR period. Everything after D can only reference D or later pictures, and D resets `frame_num` and POC.
- **K** is the forward anchor (`L1[0]`) of B1–B3. Those are the B pictures displayed before K and decoded right after it.
- **Rule 3 failed** because B1–B3 reference K, so K is not "unreferenced".

## The idea: demote D, move it into K's slot, drop K

D and K encode the same source frame. Keep D's picture data, but make it play K's role:

1. **Demote D from IDR to a non-IDR I picture (I2).**
   - `nal_unit_type` changes from 5 to 1; `nal_ref_idc` stays non-zero.
   - Slice header: remove `idr_pic_id`.
   - Replace the IDR form of `dec_ref_pic_marking` (`no_output_of_prior_pics_flag`, `long_term_reference_flag`) with the non-IDR form (`adaptive_ref_pic_marking_mode_flag = 0`).
   - This changes the header's length. With CABAC, `slice_data()` starts byte-aligned after `cabac_alignment_one_bit`, so you can rebuild the header bits (splice the bit ranges and change the fields), pad with alignment one-bits, re-escape, and then append the original slice-data bytes unchanged.
   - CABAC initialisation depends only on slice QP and slice type, which do not change. `h264hdr.parse_slice` records some bit offsets (`frame_num_bit`, `poc_lsb_bit`); extend it to record whatever the splice needs.
2. **Move I2 into K's decode slot and drop K.** The SPS and PPS in front of D can stay where they are or move with it; either is harmless. B1–B3 now decode after I2.
3. **Give I2 K's `frame_num` and K's POC.**
   - B1–B3 then see exactly the same POC distances to their anchor. Temporal direct MV scaling and implicit bi-pred weights depend only on POC differences, so they stay unchanged.
   - In D's former IDR period, shift every POC by the constant `POC(K) − POC(D)` and continue `frame_num` from K's. Write `poc_lsb = newPOC mod MaxPicOrderCntLsb` and check that the POC MSB derivation carries through correctly. These fields are fixed-width, so they can be patched in place like `frame_num` already is. Do this up to the next IDR.
   - Inter-picture POC differences inside that period are preserved, so its own direct modes and weights also stay identical.
4. **B1–B3 need repair.** They are non-reference pictures, so no error can spread beyond them.
   - **Motion field:** their temporal-direct blocks (B_Skip, B_Direct_16x16 and direct 8x8 sub-blocks) took motion from K's motion field. I2 is all-intra, so those blocks would collapse to zero motion. Rewrite every such block as explicit bi-predicted 8x8 with the *original* motion, derived with K as the co-located picture. `patch.py` / `verify_motion.reexpress` already do exactly this for B pictures whose co-located picture got PCM macroblocks; reuse that machinery.
   - **Pixels:** blocks that read K now read I2. That is two encodes of the same source, so expect compression-noise differences, plus occasional local "refresh" spots of up to about 50 levels (seen in the sample's P/P twins). Use the existing repair logic (`repair.py`: `changed_blocks`, `PixelOracle`, I_PCM patching) to patch macroblocks whose decoded result moves noticeably. Choose a sensible threshold. Noise-level differences are fine per the owner, so don't PCM entire frames.
5. **Check that D's former IDR period still decodes bit-exact.** The decoded picture buffer is no longer flushed, so older reference pictures (P0 and friends) linger:
   - **P lists** sort by PicNum, so the extra pictures land at the end. **B lists** sort by POC: in L0 they come after I2, and in L1 they come after the entries that follow the current picture. Every reference index actually used should therefore still point at the same picture.
   - **Edge case:** the rule "if L1 has more than one entry and equals L0, swap the first two" can now fire where it didn't before. Compute the new lists with `h264mv.dpb_walk` on the rewritten stream and compare them against the old lists for every index in use. If one differs, fix it with ref_pic_list_modification or fall back to the substitute for that pair.
   - Sliding-window marking with `max_num_ref_frames = 4` evicts the oldest pictures first, so they age out within a few pictures.
6. **Keep the seek point.** I2 is no longer an IDR. Insert a recovery point SEI before I2's slice (payloadType 6: `recovery_frame_cnt = 0`, `exact_match_flag = 1`, `broken_link_flag = 0`, `changing_slice_group_idc = 0`) so it stays a random-access point (open GOP). B1–B3 are leading pictures that also reference P0, so they can't decode after a seek straight to I2, which is normal open-GOP behaviour.
   - Check that mkvmerge flags the I2 access unit as a keyframe (`ffprobe -show_packets` flags, or mkvinfo).
   - Check that seeking to it works, e.g. by decoding a segment that starts at I2 with ffmpeg.
   - If mkvmerge won't flag it, report that; don't silently ship broken seeking.
7. **Integrate as a new drop rule** taking the place of rule 6 (visible substitute). Apply it when all of these hold:
   - the later twin is an IDR;
   - the earlier twin K is referenced only by pictures that the rebuild repairs (normally just B1–B3);
   - the list check in step 5 passes.

   Otherwise fall back to the current behaviour, and report how often that happens. Add report fields (e.g. `anchor_rebuilds`: per pair, the display index, PCM MBs, and the worst difference in B1–B3) and a README section. Exactly one frame per pair is dropped, the duplicate itself, so the A/V frame accounting is unchanged.

## Acceptance

- **smoke.mkv `--verify`:** 0 visible substitutes (or a reported, justified handful of fallbacks).
  - Every picture outside B1–B3 of each rebuilt pair is bit-exact. This includes the whole former-IDR period after I2.
  - Report the worst difference and the macroblocks differing by 3 or more inside B1–B3.
  - The existing hard-drop neighbourhoods behave as before.
- **A/V:** the video/audio duration check still passes. Sync should improve now that there are no substitutes.
- **Seeking:** check it near a few rebuilt pairs and report what you saw.
- **Regression:** foliage and sample have no `PIi` substitutes. Their outputs should be byte-identical to the current ones, or explain any difference. Re-run `--verify` on them.
- **Tests:** add a focused test (e.g. a small segment around smoke pair display 720 that rebuilds and decodes) and keep `tests/` passing.

## Practical notes

- **The baseline to compare against** is kept in `work/smoke_out.mkv` and `work/smoke_report.json` (current tool, with substitutes).
- **Iterate on one pair first.** Decode just the IDR period(s) around it with a segment decode (see `frame_dedup/decode.py` and the verify module), compare by POC against the original, and only then run the full file.
- **Validated modules:** `h264cabac.py`, `h264mv.py` and `patch.py` are validated (byte-exact CABAC round trip, bit-exact motion re-expression). If you change them, re-run `frame_dedup/roundtrip_test.py` and `frame_dedup/verify_motion.py` on a few pictures.
- **The owner's PC has crashed under full load.** The cause was an AMD GPU driver bug: bugcheck 0x133 inside `amdkmdag`, during an MPO flip. Keep the tool's throttled defaults (half the cores, capped ffmpeg decoders, below-normal priority). Run one heavy job at a time, and flush logs as you go.
- **Environment:** Windows 11 with Git Bash or PowerShell. Python 3.12 with numpy and Pillow only. No C compiler, no numba, do not pip-install anything. ffmpeg, ffprobe, mkvmerge and mkvextract are on PATH. Disk is tight: delete large intermediates when done.
- **Git:** commit in logical steps with `-c user.name="aquatakat" -c user.email="aquatakat@gmail.com"`, ending every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never commit media or intermediates (`.gitignore` covers them).
