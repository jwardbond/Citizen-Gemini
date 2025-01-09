# Making Large Legislative Records Accessible with AI

## The Challenge
- Ontario Legislature has massive amounts of transcript data
- 20M+ tokens of parliamentary proceedings
- Too large to fit in LLM context windows
- Need efficient way to find and load relevant content

## Our Solution: Smart Context Management

We use two different LLM models: one that helps to retrieve relevant content, and one that answers the question. We leverage Gemini's context caching to cut down on costs and compute time. See `docs.md` for more details.