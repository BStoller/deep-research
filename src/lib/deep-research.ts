import FirecrawlApp, { SearchResponse } from "@mendable/firecrawl-js";
import { generateObject, generateText } from "ai";
import { compact } from "lodash-es";
import pLimit from "p-limit";
import { z } from "zod";

import { trimPrompt, mini4oModel, deepseekR1Model } from "../ai/providers";
import { systemPrompt } from "./prompt";

type ResearchResult = {
  learnings: string[];
  visitedUrls: string[];
};

// increase this if you have higher API rate limits
const ConcurrencyLimit = 2;

// Initialize Firecrawl with optional API key and optional base url

const firecrawl = new FirecrawlApp({
  apiKey: process.env.FIRECRAWL_KEY ?? "",
  apiUrl: process.env.FIRECRAWL_BASE_URL,
});

// take en user query, return a list of SERP queries
async function generateSerpQueries({
  query,
  numQueries = 3,
  learnings,
}: {
  query: string;
  numQueries?: number;
  learnings?: string[];
}) {
  const response = await generateText({
    model: deepseekR1Model,
    system: systemPrompt(),
    prompt: `Given the following prompt from the user, generate a list of SERP queries to research the topic. Return exactly ${numQueries} queries    
    Make sure each query is unique and not similar to each other.
    Here are some learnings from previous research: ${learnings?.join("\n")}
    User prompt: ${query}`,
  });

  const data = await generateObject({
    model: mini4oModel,
    system: systemPrompt(),
    prompt: response.text,
    schema: z.object({
      queries: z.array(z.string()).describe(`List of ${numQueries} queries`),
    }),
  });

  return {
    responseText: response.text,
    queries: data.object.queries,
  };
}

async function processSerpResult({
  query,
  result,
  numLearnings = 5,
  numFollowUpQuestions = 3,
}: {
  query: string;
  result: SearchResponse;
  numLearnings?: number;
  numFollowUpQuestions?: number;
}) {
  const rawContents = compact(result.data.map((item) => item.markdown));
  const trimmedContents = rawContents.map((content) => {
    const trimmed = trimPrompt(content, 35_000);
    console.log(
      `[deep-research.ts][processSerpResult] Trimmed content length: ${trimmed.length}`
    );
    return trimmed;
  });

  const response = await generateText({
    model: deepseekR1Model,
    system: systemPrompt(),
    prompt: `Given the following contents from a SERP search for the query <query>${query}</query>, 
      generate a detailed list of ${numLearnings} learnings and ${numFollowUpQuestions} follow-up questions. 
      For each learning, provide an in-depth explanation that includes context, technical details, 
      relevant metrics, and implications.\n\n<contents>\n
      ${trimmedContents
        .map((content) => `<content>\n${content}\n</content>`)
        .join("\n")}\n</contents>`
  });

  const data = await generateObject({
    model: mini4oModel,
    system: systemPrompt(),
    prompt: response.text,
    schema: z.object({
      learnings: z
        .array(z.string())
        .describe(`List of ${numLearnings} learnings`),
      followUpQuestions: z
        .array(z.string())
        .describe(`List of ${numFollowUpQuestions} follow-up questions`),
    }),
  });

  return {
    responseText: response.text,
    learnings: data.object.learnings,
    followUpQuestions: data.object.followUpQuestions,
  };
}

export { processSerpResult };

export async function writeFinalReport({
  query,
  prompt,
  learnings,
  visitedUrls,
}: {
  query: string;
  prompt: string;
  learnings: string[];
  visitedUrls: string[];
}) {
  const learningsString = trimPrompt(
    learnings
      .map((learning) => `<learning>\n${learning}\n</learning>`)
      .join("\n"),
    175_000
  );

  // Updated prompt: instruct the LLM to generate a much longer report (at least 5 pages)

  const finalPrompt = `Given the following prompt from the user, write an extremely detailed final report on the topic using the learnings from research. The report should be highly comprehensive—aim for at least 5 pages of detailed analysis when formatted. For each learning provided below, elaborate on its implications with extended discussions that include multiple case studies, data analysis, and, where applicable, citations or references to support the findings. Provide an in-depth executive summary at the top that reiterates the original query and addresses any follow-up questions, provide a table of contents if required, then expand and provide detailed sub sections.  Include detailed technical analysis, comparisons, and examples throughout the report. Reference case studies and provide citations if appropriate\n\n<prompt>${prompt}</prompt>\n\nHere are all the learnings from previous research:\n\n<learnings>\n${learningsString}\n</learnings>\n\nEnsure that you include the original query as follows:\n\n<query>\n${query}\n</query>\n\nTake into account the original intent of the primary query and the follow-up questions. More detail is better. Write in markdown format.`;

  console.log(
    "[deep-research.ts][writeFinalReport] Prompt sent to LLM:",
    finalPrompt
  );

  const res = await generateText({
    model: deepseekR1Model,
    system: systemPrompt(),
    prompt: finalPrompt,
  });

  console.log("[deep-research.ts][writeFinalReport] LLM response:", res.text);

  // Append the visited URLs section to the report
  const urlsSection = `\n\n## Sources\n\n${visitedUrls
    .map((url) => `- ${url}`)
    .join("\n")}`;
  return res.text + urlsSection;
}
export async function deepResearch({
  query,
  breadth,
  depth,
  learnings = [],
  visitedUrls = [],
}: {
  query: string;
  breadth: number;
  depth: number;
  learnings?: string[];
  visitedUrls?: string[];
}): Promise<ResearchResult> {
  const serpMessages = await generateSerpQueries({
    query,
    numQueries: breadth,
    learnings,
  });

  const limit = pLimit(ConcurrencyLimit);

  const results = await Promise.all(
    serpMessages.queries
      .slice(-breadth) // Take only the last 'breadth' number of messages (AI responses)
      .map((serpMessage) =>
        limit(async () => {
          try {
            const result = await firecrawl.search(serpMessage, {
              timeout: 15000,
              limit: 5,
              scrapeOptions: {
                formats: ["markdown"],
                onlyMainContent: true,
                waitFor: 3000,
                removeBase64Images: true,
              },
            });

            const newUrls = compact(result.data.map((item) => item.url));
            const newBreadth = Math.ceil(breadth / 2);
            const newDepth = depth - 1;

            const { learnings: _learnings, followUpQuestions } = await processSerpResult({
              query: serpMessage,
              result,
              numLearnings: 5,
              numFollowUpQuestions: newBreadth,
            });

            const allLearnings = [...learnings, ..._learnings];
            const allUrls = [...visitedUrls, ...newUrls];

            if (newDepth > 0) {
              console.log(
                `[deep-research.ts] Researching deeper for query "${serpMessage}", breadth: ${newBreadth}, depth: ${newDepth}`
              );

              const followUpMessages = followUpQuestions.slice(-newBreadth); // Take only the AI responses
              const nextQuery = `Previous research goal: ${query}\nFollow-up research directions: ${followUpMessages.join(
                "\n"
              )}`;

              return deepResearch({
                query: nextQuery,
                breadth: newBreadth,
                depth: newDepth,
                learnings: allLearnings,
                visitedUrls: allUrls,
              });
            } else {
              return {
                learnings: allLearnings,
                visitedUrls: allUrls,
              };
            }
          } catch (e: any) {
            if (e.message && e.message.includes("Timeout")) {
              console.error(
                `[deep-research.ts] Timeout error running query: "${serpMessage}":`,
                e
              );
            } else {
              console.error(
                `[deep-research.ts] Error running query: "${serpMessage}":`,
                e
              );
            }
            return { learnings: [], visitedUrls: [] };
          }
        })
      )
  );

  return {
    learnings: [...new Set(results.flatMap((r) => r.learnings))],
    visitedUrls: [...new Set(results.flatMap((r) => r.visitedUrls))],
  };
}
