# Geodd

**Fast, steady inference for AI agents.**

Geodd runs production inference and uses real workloads to improve the code that executes it.

Our AI agents, powered by hardware-specific LLMs, identify execution bottlenecks, write kernel code, test changes, and deploy verified improvements back into serving. Successful kernels and performance measurements guide the next optimization cycle.

Use the inference service through our API, or evaluate our publicly released optimized runtimes on your own GPUs.

[Website](https://geodd.io) · [Model Engine](https://geodd.io/model-engine) · [Optimized Runtimes](https://geodd.io/optimized-models) · [Documentation](https://geodd.io/docs) · [Model Catalog](https://geodd.io/models)

## AI agents that write inference kernels

Production traffic exposes execution conditions that matter: changing context lengths, batch sizes, concurrency, and the time spent in different parts of inference.

We use those conditions to guide kernel development and runtime optimization.

| Stage | What happens |
| --- | --- |
| Measure | Observe runtime behavior and identify where inference spends time. |
| Develop | Hardware-specific AI agents generate and refine kernel code for the bottleneck. |
| Verify | Test correctness and measure performance under the workload conditions that exposed the problem. |
| Deploy | Introduce verified improvements into the inference service and measure their production effect. |
| Improve the next iteration | Use successful kernel code and measured results to inform subsequent experiments and further fine-tuning of our specialist LLMs. |

The work includes source-level kernel changes and corrections to how a runtime executes a model. The output is an implementation that can be built, tested, and deployed.

**Our optimization agents use runtime performance measurements—not the text of customer prompts or completions.**

Read about the [model engine](https://geodd.io/model-engine) and how improvements return to the [inference service](https://geodd.io/inference-service).

### Hardware-specific LLMs

These are the models behind Geodd’s optimization agents. They are separate from the application models available through our inference API.

| Specialist | Hardware and software stack | Public information |
| --- | --- | --- |
| **Mosaic** | NVIDIA GPUs and CUDA | [Benchmarks and recorded results](https://geodd.io/model-engine/mosaic) |
| **Druze** | AMD GPUs and ROCm | Coming soon |
| **Strata** | Tenstorrent accelerators and TT-Metalium | Coming soon |

## What the agents have changed

### Qwen2.5-7B: source-level optimization on H200

For our Qwen2.5-7B implementation, a Mosaic-powered agent analyzed the workload, selected a TensorRT-LLM execution path, and **modified the GEMM plugin source**. It combined those changes with FP8 execution, context-attention settings, and a paged KV-cache configuration.

We rebuilt and benchmarked the implementation, then packaged it for evaluation by Phala’s engineering team.

**Published workload:** Qwen2.5-7B · 1× NVIDIA H200 · FP8 · 2,000 input tokens · 1,000 output tokens · 32 concurrent requests.

| Metric | vLLM baseline | Geodd implementation |
| --- | ---: | ---: |
| Mean generation rate per request | 128.57 tokens/s | 195.16 tokens/s |
| Change in mean generation rate | — | +51.8% |
| Change in mean time to first token | — | 23.8% lower |

These are **per-request generation rates with 32 requests in flight**, not aggregate server throughput.

The comparison measures complete serving implementations. It does not isolate the GEMM source changes from the runtime and other execution settings, and it is not a performance guarantee for other workloads.

[Read the Qwen2.5-7B case study](https://geodd.io/case-studies/qwen2-5-7b-inference-optimisation-with-phala).

### Trinity Mini: correcting the attention execution path

When bringing Arcee’s Trinity Mini to TensorRT-LLM, our agent identified missing per-layer attention-window configuration. The runtime was applying full attention to every layer instead of respecting the model’s combination of sliding-window and global attention.

The agent produced a runtime correction that supplied the appropriate configuration. This addressed the serving path, not the model weights.

[Read the Trinity Mini engineering write-up](https://geodd.io/case-studies/bringing-arcee-trinity-to-tensorrt-llm-part-one).

## Evaluate optimized runtimes on your own GPUs

We publish free Docker runtimes containing model-specific kernels developed by our AI agents. The releases package inference implementations using vLLM, SGLang, or TensorRT-LLM, depending on the runtime.

| Runtime | Published Docker image |
| --- | --- |
| Qwen 2.5 7B | `geodd/qwen-2.5-7b:latest` |
| Llama 3.3 70B | `geodd/llama-3.3-70b:latest` |
| Mistral Nemo | `geodd/mistral-nemo:latest` |
| Qwen3 32B | `geodd/qwen3-32b:latest` |
| Trinity Mini | `geodd/trinity-mini:latest` |

Open the [optimized-runtime catalog](https://geodd.io/optimized-models) for the published Docker instructions.

The listed releases require an NVIDIA GPU, a compatible driver, and NVIDIA Container Toolkit. Check the requirements for the selected runtime before running it.

These are inference-runtime releases, **not downloads of Mosaic or our other internal kernel-development LLMs**. Performance depends on the hardware, workload, and configuration; an image tag alone does not establish benchmark reproducibility.

## Use Geodd in your application

Geodd offers an OpenAI-compatible interface for supported language-model endpoints. The following examples use [`openai/gpt-oss-120b`](https://geodd.io/models/openai/gpt-oss-120b).

Create an API key through the Geodd console using the [quick-start guide](https://geodd.io/docs/llm-api/getting-started), then configure `GEODD_API_KEY` in your server environment or secret manager.

**Keep credentials server-side. Do not place them in browser code, public environment variables, source control, or logs. Real inference requests incur usage charges.**

### Python

Install the SDK:

```bash
python -m pip install openai
```

```python
import os

from openai import APIError, OpenAI

api_key = os.environ.get("GEODD_API_KEY")
if not api_key:
    raise SystemExit("Set GEODD_API_KEY in your server environment.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.geodd.io/inference/v1",
)

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": "Explain the difference between prefill and decoding.",
            }
        ],
    )
except APIError as error:
    # Report the error type without exposing credentials or request bodies.
    raise SystemExit(
        f"Geodd request failed: {type(error).__name__}"
    ) from None

if not response.choices:
    raise SystemExit("The endpoint returned no completion choices.")

print(response.choices[0].message.content or "")
```

### TypeScript / Node.js

Install the SDK with your project’s package manager:

```bash
npm install openai
```

```typescript
import OpenAI from "openai";

const apiKey = process.env.GEODD_API_KEY;

if (!apiKey) {
  throw new Error("Set GEODD_API_KEY in your server environment.");
}

const client = new OpenAI({
  apiKey,
  baseURL: "https://api.geodd.io/inference/v1",
});

async function main(): Promise<void> {
  const response = await client.chat.completions.create({
    model: "openai/gpt-oss-120b",
    messages: [
      {
        role: "user",
        content: "Explain the difference between prefill and decoding.",
      },
    ],
  });

  if (!response.choices.length) {
    throw new Error("The endpoint returned no completion choices.");
  }

  console.log(response.choices[0].message.content ?? "");
}

main().catch((error: unknown) => {
  if (error instanceof OpenAI.APIError) {
    console.error(
      `Geodd request failed: ${error.status ?? "connection error"}`,
    );
  } else {
    console.error("The integration failed before a usable response was returned.");
  }

  process.exitCode = 1;
});
```

For production applications, configure timeouts, retry behavior, and error handling to suit the workload. See the [chat-completion reference](https://geodd.io/docs/llm-api/chat-completion) and the SDK documentation for [Python](https://github.com/openai/openai-python) or [TypeScript](https://github.com/openai/openai-node).

### Regional API routes

The GPT-OSS-120B page currently lists these serving routes:

| Serving region | API base URL |
| --- | --- |
| United States | `https://api.geodd.io/inference/v1` |
| Europe — Stavanger, Norway | `https://eu.api.geodd.io/inference/v1` |

Change `base_url` in Python or `baseURL` in TypeScript to select the applicable route.

Availability is model-specific. The European route’s published location is Norway; the `eu` hostname should not be interpreted as an EU-only residency commitment. Review the selected model’s [regional information](https://geodd.io/models/openai/gpt-oss-120b#regions), data-handling policy, and applicable agreement.

## Integrate with your coding agent

Give the following instructions to your coding agent to adapt an existing application. This is an application-integration workflow, separate from Geodd’s own kernel-development agents.

```text
Integrate Geodd inference into this application.

First inspect the existing codebase. Preserve its architecture,
package manager, server-side boundaries, and integration patterns.

Use:
- Model: openai/gpt-oss-120b
- API base URL: https://api.geodd.io/inference/v1
- Credential environment variable: GEODD_API_KEY
- Chat-completions interface through a compatible OpenAI SDK

Read the current documentation and model facts before implementing:
https://geodd.io/docs/llm-api/getting-started
https://geodd.io/models/openai/gpt-oss-120b.json

Keep credentials server-side. Never request a secret in this conversation
or expose it through browser code, public variables, logs, or commits.

Preserve conversation history where applicable. Implement streaming,
tool calling, or structured outputs only when required by the application
and supported by the endpoint. Treat unknown model facts as unverified.

Handle authentication failures, rate limits, connection failures, and
upstream errors. Follow the project's testing conventions.

Ask for approval before making billable requests. Without authorized
credentials and billing approval, use mocks and clearly identify what
has not been tested against the live service.

Summarize changed files, configuration requirements, and actual test results.
```

The Geodd CLI is currently described as **coming soon**. Use the documented SDK workflow rather than assuming a Geodd installation or authentication command exists.

### Machine-readable model facts

The [GPT-OSS-120B JSON representation](https://geodd.io/models/openai/gpt-oss-120b.json) exposes model identifiers, pricing, region details, and published endpoint capabilities.

An unknown field is represented as `null`, not as a guarantee of support or a declaration that the feature is unsupported. Catalog information is not a substitute for testing your application against the endpoint.

## Model catalog

The following links lead to individual Geodd model pages, where developers can review current pricing, model details, and available integration information.

**Catalog checked: September 20, 2026.** Use the [live model catalog](https://geodd.io/models) for subsequent additions and changes.

### Text models

| Model | Model ID |
| --- | --- |
| [GLM 5.2](https://geodd.io/models/zai-org/glm-5.2) | `zai-org/glm-5.2` |
| [DeepSeek V4 Flash](https://geodd.io/models/deepseek-ai/DeepSeek-V4-Flash) | `deepseek-ai/DeepSeek-V4-Flash` |
| [GPT-OSS-120B](https://geodd.io/models/openai/gpt-oss-120b) | `openai/gpt-oss-120b` |
| [Kimi K2.6](https://geodd.io/models/moonshotai/Kimi-K2.6) | `moonshotai/Kimi-K2.6` |
| [DeepSeek V4 Pro](https://geodd.io/models/deepseek-ai/DeepSeek-V4-Pro) | `deepseek-ai/DeepSeek-V4-Pro` |
| [Gemma 4 31B](https://geodd.io/models/google/gemma-4-31B-it) | `google/gemma-4-31B-it` |
| [GPT-OSS-20B](https://geodd.io/models/openai/gpt-oss-20b) | `openai/gpt-oss-20b` |
| [Seed 2.1 Turbo](https://geodd.io/models/bytedance/dola-seed-2-1-turbo-260628) | `bytedance/dola-seed-2-1-turbo-260628` |
| [Seed 2.0 Pro](https://geodd.io/models/bytedance/seed-2-0-pro-260328) | `bytedance/seed-2-0-pro-260328` |

### Image models

| Model | Model ID |
| --- | --- |
| [Seedream 5.0 Pro](https://geodd.io/models/bytedance/dola-seedream-5-0-pro-260628) | `bytedance/dola-seedream-5-0-pro-260628` |
| [Seedream 5.0 Lite](https://geodd.io/models/bytedance/seedream-5-0-260128) | `bytedance/seedream-5-0-260128` |

### Video models

| Model | Model ID |
| --- | --- |
| [Seedance 2.0](https://geodd.io/models/bytedance/dreamina-seedance-2.0) | `bytedance/dreamina-seedance-2.0` |
| [Seedance 2.0 Fast](https://geodd.io/models/bytedance/dreamina-seedance-2.0-fast) | `bytedance/dreamina-seedance-2.0-fast` |

Capabilities and request formats vary by model. Do not assume that every endpoint supports the same streaming, tool-calling, structured-output, or deployment options, or that image and video models use chat completions.

The hosted catalog and the [optimized-runtime releases](https://geodd.io/optimized-models) are separate. A hosted model listing is not a claim that a custom kernel release or benchmark has been published for that model.

## Zero Data Retention and data handling

**Production-driven optimization does not mean training on your prompts.**

Under Geodd’s default [API data-handling policy](https://geodd.io/trust-center/data-handling), prompts, inputs, outputs, and request and response bodies are processed transiently rather than stored, unless a different arrangement is separately agreed in writing.

Customer API content is not used for model training or fine-tuning by default. Human review of prompts and outputs is also not performed by default.

Limited operational metadata may be retained, including model identifiers, timestamps, token counts, status codes, and usage records, for billing, security, troubleshooting, and service operation.

**Zero payload retention does not mean zero metadata retention.** Service configuration, customer-controlled infrastructure, and separately agreed arrangements can have different responsibilities and requirements.

[Data Handling](https://geodd.io/trust-center/data-handling) · [Privacy Policy](https://geodd.io/legal/privacy-policy) · [Security Measures](https://geodd.io/trust-center/security-measures)

## Deployment options and developer resources

| Resource | What it provides |
| --- | --- |
| [Serverless and dedicated inference](https://geodd.io/inference-service) | Managed API access and isolated inference deployment options. Confirm model and workload requirements for dedicated deployments. |
| [Dedicated GPU infrastructure](https://geodd.io/dedicated-deployment) | Dedicated hardware with customer responsibility for the operating system, drivers, and runtime. This is distinct from a managed inference endpoint. |
| [API documentation](https://geodd.io/docs) | Integration guides and endpoint references. |
| [Pricing](https://geodd.io/pricing) | Published pricing and deployment information. |
| [Mosaic benchmarks](https://geodd.io/model-engine/mosaic) | Recorded kernel-development and inference optimization results. |
| [Engineering case studies](https://geodd.io/case-studies) | Workloads, agent-produced changes, and measured outcomes. |
| [Service status](https://status.geodd.io) | Operational status information. |

Working on an inference bottleneck, evaluating a runtime, or planning a deployment? [Talk to Geodd’s engineering team](https://geodd.io/contact).
