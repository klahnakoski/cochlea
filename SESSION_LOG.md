# Session log

## 2026-10-19
- suite 33 passed (pytest), web/frames.test.mjs ok, on dev 1c027b4
- built since repo creation: gammatone filterbank spectrogram (one FFT, 96 bins), ISO 226 loudness, 3 dB/oct emphasis, harmonic suppress, kick flash, hits/top/notes views, default music folder with track picker, spiral + flat strip chart
- Notes view (cochlea/notes.py): rectangles on a 1/4 s Gaussian STFT; merged voice branch (8737d63): chained short steps, second quieter voice pass, tones kept known through beats; page hides the voice pass (SHOW_VOICE false) because it mostly finds quiet held tones
- scripts/rectangles.py mask panel now shows known pixels with beats black; latest charts in charts/mask2
- half-done, branch glide 136cedb (worktree .claude/worktrees/glide, not merged): voice pass as parallelograms on slope-smeared thin-line contrast, bass smoothing below C3; 32 pass, 1 fail (test_quiet_vibrato_shows_in_the_voice_pass), 1 xfail; +1.1 s per song; "mine" not found, bass still dashes
- next: ask the user where "mine" is (E4..A4 vs E5..G5 at 1:06.6), then finish or replace glide
