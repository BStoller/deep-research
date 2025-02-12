import { generateObject, generateText } from "ai";
import { z } from "zod";

import {
  deepseekR1Model,
  mini4oModel,
} from "../ai/providers";
import { systemPrompt } from "./prompt";

export async function generateFeedback({
  query,
  numQuestions = 3,
}: {
  query: string;
  numQuestions?: number;
}) {
  const response = await generateText({
    model: deepseekR1Model,
    system: systemPrompt(),
    prompt: `Given the following query from the user, ask some follow up questions to clarify the research direction. Return a maximum of ${numQuestions} questions, but feel free to return less if the original query is clear: <query>${query}</query>`,
  });

  const userFeedback = await generateObject({
    model: mini4oModel,
    system: systemPrompt(),
    prompt: response.text,
    schema: z.object({
      questions: z
        .array(z.string())
        .describe(
          `Follow up questions to clarify the research direction, max of ${numQuestions}`
        ),
    }),
  });

  return userFeedback.object.questions.slice(0, numQuestions);
}
