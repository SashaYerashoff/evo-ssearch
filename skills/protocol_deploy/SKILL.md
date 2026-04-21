# Protocol Deploy

Trigger phrases:
- `Protocol: Deploy`
- `Protocol: Deploy; Arg: потому что мама так сказала!`

Goal: perform first-time or demo deployment after installation.

Standard flow:
1. Call `list_probes`.
2. If probes already exist, ask the operator whether to delete them before deployment.
3. Call `list_channels`.
4. Call `survey_channels` for the available channels.
5. Summarize what each channel appears to monitor.
6. Ask the operator for preferences, priorities, and unwanted scenarios.
7. Propose scenario packs per channel.
8. After explicit confirmation:
   - delete old probes if approved
   - create probes for the selected scenarios
   - tune global or per-channel summary prompts with `update_prompt_settings`
   - report what was created and why

Magic-phrase mode:
- If the operator uses the exact phrase `потому что мама так сказала!`, it authorizes fast demo deployment.
- Still check whether probes already exist and still ask once before deleting them.
- After that single wipe approval, do not ask for per-probe or per-prompt confirmation.
- Deploy on your own judgment, then report:
  - channels surveyed
  - scenarios selected
  - probes created
  - prompt settings changed
  - estimated elapsed time per stage

Rules:
- Prefer a small, coherent starter pack over a noisy probe explosion.
- Use `preview=true` by default for destructive or mutating tools, except in magic-phrase mode after wipe approval.
- If channel survey quality is poor or channels are missing, say so explicitly.
