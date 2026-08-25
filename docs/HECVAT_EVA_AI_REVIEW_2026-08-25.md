# EVA AI HECVAT AI Controls Review

Review date: 2026-08-25

Product version: EVA AI 0.8.7 trusted pilot

Source workbook: `HECVAT416_Luxriot_Reviewed_FINAL.xlsx`

## Quick-entry table

| # | Control | Answer | Paste-ready explanation | Owner |
|---:|---|:---:|---|---|
| 25 | AIGN-03 | No | Formal responsible AI training has not yet been completed for all personnel developing or supporting EVA AI. During the 0.8.7 trusted pilot, the EVA AI team uses documented threat-model, privacy, human-oversight, model-limitation, and evidence-grounding guidance during engineering review; however, this is not represented as a completed training program. A role-based responsible AI training program and completion records should be added to the path toward version 1.0. | Riot Engineering Ltd organizational HR/Compliance, supported by EVA AI Engineering |
| 27 | AIGN-05 | No | EVA AI 0.8.7 does not implement configurable content-DLP business rules that inspect and block sensitive prompt, image, or query content before it reaches the configured AI model. Compensating controls include on-premises processing, RBAC and channel grants, PostgreSQL row-level security, bounded retention, server-side disabling of selected ingestion features, file type/size/path validation, and secret redaction in audit and agent-tool output. These controls limit access and exposure but are not a DLP system. | EVA AI Engineering |
| 30 | AIPL-02 | No | EVA AI-specific risks have been identified qualitatively, but they have not yet been measured through a formal AI risk register or risk-management framework. Documented risks include prompt injection and excessive agency, false or missed detections, incomplete visual coverage under load, privacy and retention risk, insecure external service links, third-party model and dependency risk, and failed external alert delivery. The current pilot has no documented likelihood/impact scoring, formal risk acceptance, named organizational risk owners, or corporate liability process. | EVA AI Engineering and Riot Engineering Ltd organizational Risk/Legal owners |
| 36 | AISC-02 | No | No, under the training and model-improvement meaning defined by this question. Customer prompts, camera frames, uploads, searches, alert criteria, and queries are used for request-time inference and local product operation. They are not used to fine-tune or update Qwen or SigLIP model weights, personalize shared model weights, or train across customers. EVA may retain local evidence, embeddings, summaries, agent history, and operator annotations under configured retention controls. Feedback may support human-reviewed prompt or probe proposals but does not automatically train the model. | EVA AI Engineering; contractual commitments require Legal confirmation |
| 39 | AISC-05 | No | EVA AI 0.8.7 has deterministic offline dependency manifests, pinned model checkpoints, SHA-256 release-payload verification, reproducible installation assets, and functional CI tests. However, it does not yet maintain a standards-based AI SBOM or run automated dependency, SAST, secret, container, or model-vulnerability scanning as a release gate. Several exact model-license and provenance checks also remain open. | EVA AI Engineering and organizational Security/Legal |
| 42 | AIML-02 | No | EVA validates configured model identity and runtime health and treats model output as untrusted through structured contract parsing, bounded corrective retries or deterministic fallbacks, evidence linkage, and operator review. Authenticated operator annotations are linked to the exact detection and are not used to retrain model weights. However, EVA AI 0.8.7 has no formal model-feedback authentication program, poisoning/skewing/date-or-phrase attack evaluation, or trigger-sweep testing process. | EVA AI Engineering |
| 46 | AIML-06 | Yes | EVA uses model-defense mechanisms rather than claiming adversarial training. Inputs are validated and bounded by supported type, size, path, schema, image count, and context size. Model output is parsed and cannot directly execute actions. Agent tools are allow-listed and protected by server-side permissions, channel scope, argument schemas, resource bounds, audit, and human approval. SigLIP runtime canaries fail closed on embedding drift, and content-aware VLM health checks can quarantine unhealthy inference. | EVA AI Engineering |
| 47 | AIML-07 | Yes | EVA provides operational model transparency through documentation of model profiles, prompts, evidence hierarchy, known limitations, sampling coverage, and human-review requirements. Agent messages and audited tool actions are retained locally. VLM records retain model output, structured batch state, prompt or bounded prompt excerpt, runtime statistics, sampled evidence, and channel/time/batch provenance. Access is protected through authentication, RBAC, channel grants, PostgreSQL row-level security, and configurable retention. This is operational traceability, not full internal model interpretability. | EVA AI Engineering and customer data administrator |
| 48 | AIML-08 | N/A | EVA AI does not create or fine-tune an ML training dataset in customer deployments. It uses third-party pretrained checkpoints and pins shipped model files by version and cryptographic hash. EVA does not watermark upstream training data and cannot claim that the original model provider's training corpus is watermarked. If the workbook does not accept N/A, use No with this explanation. | EVA AI Engineering; upstream provenance and licensing require model-provider/Legal evidence |
| 50 | AILM-01 | Yes | EVA limits LLM privileges by default. The LLM cannot directly access the database, filesystem, operating-system shell, application credentials, or arbitrary network endpoints. It may request only registered server-side tools exposed through an intent- and workflow-specific allow-list. The gateway derives authorization context server-side and enforces permission, channel scope, argument schemas, resource bounds, rate limits, timeouts, audit, and one-time human approval for sensitive actions. | EVA AI Engineering |
| 51 | AILM-02 | No | EVA AI does not train or fine-tune the bundled Qwen LLM and does not maintain a first-party LLM training corpus. Engineering verifies the identity and hash of the shipped checkpoint and tracks model licensing, but cannot attest that the upstream Qwen training dataset has been fully vetted for provenance, licensing, PII, bias, safety, or refresh history. Customer data is not used for LLM training. | EVA AI Engineering; upstream dataset claims require model-provider/Legal evidence |
| 52 | AILM-03 | Yes | Read-only research and status operations may execute within the authenticated user's scope. Sensitive or behavior-changing actions require human intervention. Probe and prompt changes, bookmark creation, summary restoration, deployment application, and incident-state changes use preview and approval workflows. Approvals are bound to actor, tenant, action, normalized argument hash, and expiry; they are one-time use and audited. | EVA AI Engineering and authorized customer operators |
| 53 | AILM-04 | Yes | EVA does not expose a third-party plugin marketplace or arbitrary executable plugins to the LLM. The callable surface is a closed server-side tool registry. For each operator input, the model receives only the subset associated with the detected intent or active workflow. Per-turn tool budgets, duplicate-read suppression, schema validation, permissions, channel scope, timeouts, rate limits, and output bounds limit tool chaining. | EVA AI Engineering |
| 54 | AILM-05 | Yes | EVA applies bounded resource controls at multiple levels: context and output-token budgets, maximum VLM images per request, per-turn tool-call budgets, per-tool row/time/output limits and timeouts, actor-level tool rate limits, per-profile maximum inference concurrency, bounded queues, and priority-based LM admission. Host-level CPU or memory quotas per tenant are not currently implemented and remain an appliance-capacity control. | EVA AI Engineering and customer infrastructure owner |
| 55 | AILM-06 | Yes | EVA uses model-validation mechanisms rather than training on customer data. These include deterministic unit and contract tests, structured-output parsing, semantic guards, bounded corrective retries and fallbacks, content-aware VLM and embedding canaries, evidence-linked human review, and a supervised live-agent scenario harness that records pass/fail status, latency, queue behavior, tool use, generation-quality signals, and tool efficiency. Formal adversarial red-team testing and customer-scene bias/accuracy reports remain gaps. | EVA AI Engineering |
| 56 | AILM-07 | No | EVA preserves source and provenance metadata for frames, summaries, alerts, tool results, and audited actions, and separates model-visible content from server-owned authorization and approval context. It does not implement end-to-end taint labels propagated through every LLM tool input, output, and downstream content. Prompt-injection impact is constrained through the allow-listed gateway, RBAC, channel scope, bounded execution, and human approval, but these safeguards are not equivalent to taint tracing. | EVA AI Engineering |

## General product and audit information

EVA AI is a separate on-premises product and integration developed by a dedicated
team within Riot Engineering Ltd. Version 0.8.7 is a trusted pilot rather than the
formal 1.0 security release. This review describes controls implemented in the
EVA AI repository and deployment packages; it does not make unsupported claims
about corporate policy, personnel training, contractual liability, or customer
infrastructure.

Luxriot Evo is a separately developed VMS/NVR. EVA AI connects to the ordinary
partner-facing Luxriot Evo HTTP API available to integration partners. The EVA AI
team owns its client-side integration, credential handling, request behavior, and
processing of retrieved evidence. The implementation and operational security of
Luxriot Evo itself belong to the Luxriot Evo team and must not be attributed to
EVA AI Engineering.

## Current 0.8.7 control posture

Implemented controls relevant to these questions include:

- on-premises inference and storage with no cross-customer training loop;
- named users, RBAC, per-channel grants, tenant isolation, and PostgreSQL RLS;
- a closed, server-owned LLM tool registry with progressive disclosure;
- server-side tool authorization independent of model decisions;
- preview and one-time approval for sensitive actions and external side effects;
- bounded tool calls, context, output, time windows, rows, queues, concurrency,
  timeouts, and inference admission;
- structured VLM/agent output contracts, validation, retries, safe fallbacks,
  evidence provenance, and explicit coverage reporting;
- local audit, agent history, VLM summaries, structured state, sampled evidence,
  and configurable retention;
- pinned offline dependencies and models, cryptographic bundle verification,
  runtime embedding canaries, and content-aware VLM health checks.

These controls reduce excessive-agency, prompt-injection, data-access, model
drift, and availability risks. They do not make model output ground truth and do
not replace operator review.

## Material gaps

The following must remain `No`, `N/A`, or explicitly qualified until additional
evidence exists:

- no completed responsible AI training program and completion register;
- no configurable content DLP for AI inputs and outputs;
- no formally measured AI risk register or organizational liability process;
- no standards-based AI SBOM or automated dependency/SAST/secret/container scan
  release gate;
- no trigger-sweep or formal feedback-poisoning/skewing evaluation program;
- no adversarial-training claim or formal adversarial robustness report;
- no first-party control over or complete vetting of upstream Qwen training data;
- no watermarking of upstream ML training data;
- no end-to-end taint tracking across every LLM-related content flow;
- no completed customer-scene bias/accuracy assessment or formal AI red-team
  report.

Planned work must not be converted to a HECVAT `Yes` until the control is
implemented and supported by retained evidence.

## Workbook issue

`AILM-07` exists in the workbook's `Questions` reference data, but it is missing
from the visible `AI` response table, which currently ends at `AILM-06`. The row
must be restored or answered manually before the workbook is submitted.

## Evidence references

The repository paths below are relative to the EVA AI product source tree
(`evo-ssearch-office-demo` at the reviewed `main` revision).

- `docs/architecture/security_threat_model.md`
- `docs/architecture/data_retention_privacy.md`
- `docs/security/ROADMAP_TO_1_0.md`
- `security/permissions.py`
- `agent_security/eva_adapter.py`
- `agent_security/gateway.py`
- `agent_security/output.py`
- `config.py`
- `lm_admission.py`
- `luxriot_connector.py`
- `scripts/eva_offline_deploy.py`
- `.github/workflows/ci.yml`
- `tests/integration/README.md`
- `docs/legal/dependency_license_audit.md`
