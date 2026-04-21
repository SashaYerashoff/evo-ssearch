# Protocol Deploy

Trigger phrases:
- `Protocol: Deploy`
- `Protocol: Deploy; Arg: потому что мама так сказала!`

Goal: perform first-time or demo deployment after installation.

Standard flow:
1. Call `list_probes`.
2. If probes already exist, ask the operator whether to delete them before deployment.
3. If the operator refuses deletion, stop the deployment flow immediately.
4. Only continue after one of these conditions is true:
   - there are no existing probes
   - the operator explicitly approved deletion
   - the operator explicitly requested a non-destructive survey-only pass
5. Call `list_channels`.
6. Call `survey_channels` for the available channels.
7. Summarize what each channel appears to monitor.
8. Ask the operator for preferences, priorities, and unwanted scenarios.
9. Propose scenario packs per channel.
10. After explicit confirmation:
   - delete old probes if approved
   - create probes for the selected scenarios
   - tune global or per-channel summary prompts with `update_prompt_settings`
   - report what was created and why

Magic-phrase mode:
- If the operator uses the exact phrase `потому что мама так сказала!`, it authorizes fast demo deployment.
- Still check whether probes already exist and still ask once before deleting them.
- If wipe approval is denied, stop the deployment flow. Do not continue surveying or creating anything unless the operator explicitly asks for survey-only behavior.
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
- If the operator only asked to check the connection, call `list_channels` and report connection status; do not resume deployment automatically.
- If channel survey quality is poor or channels are missing, say so explicitly.
