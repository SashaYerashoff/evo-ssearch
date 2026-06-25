# Protocol Deploy

Trigger phrases:
- `Protocol: Deploy`
- `Protocol: Deploy; Arg: потому что мама так сказала!`

Goal: perform first-time or demo deployment after installation.

Standard flow:
1. Call `list_channels`.
2. Check current video-description coverage/runtime with `list_video_summary_channels` for the requested pilot period when a period is known, or explain that coverage can only be assessed after summaries run.
3. If more than 8 channels are available, do not survey all channels immediately. Present the candidate list and ask the operator to choose channels or confirm broad multi-turn survey.
4. Call `survey_channels` only for selected channels or for the confirmed chunk.
   - In standard mode, use the default survey.
   - In magic-phrase mode, prefer `fast_mode=true` and chunks of at most 8 channels unless the operator explicitly asked for a deeper pass.
5. Summarize what each reviewed channel appears to monitor and list unchecked channels.
6. Ask the operator for preferences, priorities, unwanted scenarios, and alert severity policy.
7. Propose video-description prompt/alert packs per channel or channel group.
8. After explicit confirmation:
   - tune global or per-channel summary prompts with `update_prompt_settings`
   - report which video-description streams should be started or checked in the UI
   - create or change probes only if the operator explicitly requested probe-based semantic sensors
   - call `deploy_summary`
   - report what was created and why

Magic-phrase mode:
- If the operator uses the exact phrase `потому что мама так сказала!`, it authorizes fast demo deployment.
- Do not wipe or create probes by default. Only touch probes if the operator explicitly includes probes in the deployment request.
- For video-description prompt changes in magic-phrase mode, continue with preview/apply according to the deployment's secure-tool policy.
- Deploy on your own judgment, then report:
  - channels surveyed
  - scenarios selected
  - prompt settings changed
  - video-description streams to start/check
  - probes created only if explicitly requested
  - estimated elapsed time per stage
  - call `deploy_summary` before the final narrative answer

Rules:
- Prefer a small, coherent starter pack over a noisy probe explosion.
- Use `preview=true` by default for destructive or mutating tools, except where the deployment's secure-tool policy explicitly allows immediate apply.
- `create_probe` is idempotent by name+channel. Reuse the existing probe instead of creating duplicates when the same starter scenario is rerun.
- For broad deployments, work in channel chunks and keep a clear checked/unchecked channel list.
- If the operator only asked to check the connection, call `list_channels` and report connection status; do not resume deployment automatically.
- If channel survey quality is poor or channels are missing, say so explicitly.
- Current pilot center is video descriptions. Probes are optional secondary semantic sensors, not the default deployment artifact.
