import type {
  AttentionResult,
  CheckpointList,
  GenerateParams,
  ModelInfo,
  StreamToken,
  TokenizeResult,
} from "./types";

// Requests go to the same origin and are rewritten to the FastAPI backend by
// next.config.mjs (see `rewrites`). This keeps SSE streaming CORS-free.
async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(detail || `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => jsonFetch<{ status: string; trained: boolean }>("/api/health"),

  model: () => jsonFetch<ModelInfo>("/api/model"),

  checkpoints: () => jsonFetch<CheckpointList>("/api/checkpoints"),

  load: (checkpoint: string) =>
    jsonFetch<ModelInfo>("/api/load", {
      method: "POST",
      body: JSON.stringify({ checkpoint }),
    }),

  tokenize: (text: string) =>
    jsonFetch<TokenizeResult>("/api/tokenize", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  attention: (prompt: string) =>
    jsonFetch<AttentionResult>("/api/attention", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    }),
};

export interface StreamHandlers {
  onStart?: (meta: { trained: boolean; prompt: string }) => void;
  onToken?: (token: StreamToken) => void;
  onDone?: (text: string) => void;
  onError?: (err: Error) => void;
}

/**
 * Stream a generation over Server-Sent-Events. Returns an abort function.
 * We parse the SSE frames manually from a fetch stream so the request can use
 * POST (EventSource only supports GET).
 */
export function streamGenerate(
  params: GenerateParams,
  handlers: StreamHandlers,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch("/api/generate/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`stream failed: ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line.
        let sep: number;
        while ((sep = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          dispatchFrame(frame, handlers);
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        handlers.onError?.(err as Error);
      }
    }
  })();

  return () => controller.abort();
}

function dispatchFrame(frame: string, handlers: StreamHandlers) {
  let event = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;

  try {
    const parsed = JSON.parse(data);
    if (event === "start") handlers.onStart?.(parsed);
    else if (event === "token") handlers.onToken?.(parsed);
    else if (event === "done") handlers.onDone?.(parsed.text);
  } catch {
    /* ignore malformed frame */
  }
}
