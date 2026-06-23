export interface ModelInfo {
  checkpoint: string | null;
  trained: boolean;
  device: string;
  num_params: number;
  vocab_size: number;
  config: {
    d_model: number;
    n_heads: number;
    n_layers: number;
    d_ff: number;
    block_size: number;
    pos_encoding: string;
    dropout: number;
  };
}

export interface CheckpointItem {
  name: string;
  size: number;
}

export interface CheckpointList {
  checkpoints: CheckpointItem[];
  active: string | null;
}

export interface TokenInfo {
  char: string;
  id: number;
}

export interface TokenizeResult {
  tokens: TokenInfo[];
  count: number;
  vocab_size: number;
}

export interface AttentionResult {
  tokens: string[];
  n_layers: number;
  n_heads: number;
  // attention[layer][head][query][key]
  attention: number[][][][];
}

export interface GenerateParams {
  prompt: string;
  max_tokens: number;
  temperature: number;
  top_k: number | null;
  top_p: number | null;
  seed: number | null;
}

export interface StreamToken {
  i: number;
  char: string;
  id: number;
  prob: number;
}
