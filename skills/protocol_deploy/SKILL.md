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
6. If more than 8 channels are available, do not survey all channels immediately. Present the candidate list and ask the operator to choose channels or confirm broad multi-turn survey.
7. Call `survey_channels` only for selected channels or for the confirmed chunk.
   - In standard mode, use the default survey.
   - In magic-phrase mode, prefer `fast_mode=true` and chunks of at most 8 channels unless the operator explicitly asked for a deeper pass.
8. Summarize what each reviewed channel appears to monitor and list unchecked channels.
9. Ask the operator for preferences, priorities, and unwanted scenarios.
10. Propose scenario packs per channel.
11. After explicit confirmation:
   - delete old probes if approved
   - create probes for the selected scenarios
   - tune global or per-channel summary prompts with `update_prompt_settings`
   - call `deploy_summary`
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
  - call `deploy_summary` before the final narrative answer

Rules:
- Prefer a small, coherent starter pack over a noisy probe explosion.
- Use `preview=true` by default for destructive or mutating tools, except in magic-phrase mode after wipe approval.
- `create_probe` is idempotent by name+channel. Reuse the existing probe instead of creating duplicates when the same starter scenario is rerun.
- For broad deployments, work in channel chunks and keep a clear checked/unchecked channel list.
- If the operator only asked to check the connection, call `list_channels` and report connection status; do not resume deployment automatically.
- If channel survey quality is poor or channels are missing, say so explicitly.
