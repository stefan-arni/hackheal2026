// app/frontend/src/lib/voice/recorder.ts
// Thin getUserMedia + AudioWorklet PCM capture. Audio is held in memory and
// returned as a Float32Array; the caller extracts features and discards it.
// NEVER persisted, NEVER uploaded.

export interface Capture { pcm: Float32Array; sampleRate: number; }

// AudioWorklet processor source. It runs on the audio render thread and posts
// a copy of each input render-quantum (channel 0) back to the main thread; it
// writes no output, so connecting it to the destination plays only silence
// (no mic echo). Loaded from an inline Blob URL so the module stays
// self-contained — no separate build asset to ship or path to resolve in the
// Capacitor webview.
const CAPTURE_WORKLET_SRC = `
class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length) {
      // .slice() copies out of the reused render buffer before posting.
      this.port.postMessage(input[0].slice());
    }
    return true;
  }
}
registerProcessor('pcm-capture', PcmCaptureProcessor);
`;

export async function isMicAvailable(): Promise<boolean> {
  // navigator.mediaDevices is absent in non-secure contexts and SSR;
  // 'AudioContext' in window guards against Capacitor pre-browser boot;
  // 'audioWorklet' on the prototype confirms the capture path is supported
  // (AudioWorklet ships everywhere getUserMedia does in our target webviews).
  return !!(navigator.mediaDevices &&
            'getUserMedia' in navigator.mediaDevices &&
            'AudioContext' in window &&
            'audioWorklet' in AudioContext.prototype);
}

export async function record(maxSeconds: number): Promise<Capture> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext();
  const chunks: Float32Array[] = [];
  let source: MediaStreamAudioSourceNode | null = null;
  let node: AudioWorkletNode | null = null;
  try {
    // Register the capture processor from an inline module.
    const blob = new Blob([CAPTURE_WORKLET_SRC], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    try {
      await ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }

    source = ctx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ctx, 'pcm-capture', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
    });
    node.port.onmessage = (e: MessageEvent) => {
      chunks.push(e.data as Float32Array);
    };
    source.connect(node);
    // Connect to destination so the graph pulls the processor; it emits
    // silence, so there is no audible feedback from the mic.
    node.connect(ctx.destination);

    await new Promise((res) => setTimeout(res, maxSeconds * 1000));
  } finally {
    if (node) { node.port.onmessage = null; node.disconnect(); }
    if (source) source.disconnect();
    stream.getTracks().forEach((t) => t.stop());
  }

  const sampleRate = ctx.sampleRate;
  await ctx.close();
  const total = chunks.reduce((a, c) => a + c.length, 0);
  const pcm = new Float32Array(total);
  let o = 0; for (const c of chunks) { pcm.set(c, o); o += c.length; }
  return { pcm, sampleRate };
}

export function stopAndDiscard(): void { /* tracks stopped in record() */ }
