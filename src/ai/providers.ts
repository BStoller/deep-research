import { createOpenAI, type OpenAIProviderSettings } from '@ai-sdk/openai';
import { createReplicate, type ReplicateProviderSettings } from '@ai-sdk/replicate';
import { createTogetherAI } from '@ai-sdk/togetherai';
import { getEncoding } from 'js-tiktoken';

import { RecursiveCharacterTextSplitter } from './text-splitter';
import { extractReasoningMiddleware, wrapLanguageModel } from 'ai';

interface CustomOpenAIProviderSettings extends OpenAIProviderSettings {
  baseURL?: string;
}

interface CustomReplicateProviderSettings extends ReplicateProviderSettings {
  reasoningEffort?: 'low' | 'medium' | 'high';
  structuredOutputs?: boolean;
}

// Providers
const openai = createOpenAI({
  apiKey: process.env.OPENAI_KEY!,
  baseURL: process.env.OPENAI_ENDPOINT || 'https://api.openai.com/v1',
} as CustomOpenAIProviderSettings);

const replicate = createReplicate({
  apiToken: process.env.REPLICATE_API_TOKEN!,
});

const together = createTogetherAI({
  apiKey: process.env.TOGETHER_API_KEY!,
});

const customModel = process.env.OPENAI_MODEL || 'o3-mini';

export const deepseekR1Model = wrapLanguageModel({model: together('deepseek-ai/Deepseek-R1'), middleware: [
  extractReasoningMiddleware({tagName: 'think'})
]});

export const mini4oModel = openai('gpt-4o');

// Models
export const o3MiniModel = openai(customModel, {
  reasoningEffort: customModel.startsWith('o') ? 'medium' : undefined,
  structuredOutputs: true,
});

export const o1MiniModel = openai('o1-mini');

const MinChunkSize = 140;
const encoder = getEncoding('o200k_base');

// trim prompt to maximum context size
export function trimPrompt(
  prompt: string,
  contextSize = Number(process.env.CONTEXT_SIZE) || 128_000,
) {
  if (!prompt) {
    return '';
  }

  const length = encoder.encode(prompt).length;
  if (length <= contextSize) {
    return prompt;
  }

  const overflowTokens = length - contextSize;
  // on average it's 3 characters per token, so multiply by 3 to get a rough estimate of the number of characters
  const chunkSize = prompt.length - overflowTokens * 3;
  if (chunkSize < MinChunkSize) {
    return prompt.slice(0, MinChunkSize);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap: 0,
  });
  const trimmedPrompt = splitter.splitText(prompt)[0] ?? '';

  // last catch, there's a chance that the trimmed prompt is same length as the original prompt, due to how tokens are split & innerworkings of the splitter, handle this case by just doing a hard cut
  if (trimmedPrompt.length === prompt.length) {
    return trimPrompt(prompt.slice(0, chunkSize), contextSize);
  }

  // recursively trim until the prompt is within the context size
  return trimPrompt(trimmedPrompt, contextSize);
}
