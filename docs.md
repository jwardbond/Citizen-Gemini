# OLABot Technical Documentation

## TODO
- Add caching!

## Overview
OLABot is a Python-based chatbot designed to interact with Ontario Legislature (Hansard) transcripts. It uses Google's Gemini AI to provide context-aware responses about legislative proceedings.

## Overall Design

The core technical challenge is that the transcripts are quite long, in fact totalling **20 million tokens**. This makes it infeasible to load the entire corpus into memory, even with Gemini's 1 million context window. We also want to optimize for resource usage. This led us to develop a system that dynamically loads and unloads transcripts based on the user's question. First, Gemini parses every query into a set of entities: people, bills, dates, and topics. Then, Gemini determines if the current loaded transcripts are relevant to the query by comparing these entities to transcript summaries we generated, or if the current question is a follow up to the previous question. If not, we find a new set of transcripts that are relevant and load them in. Once a set of relevant transcripts are loaded, we finally use Gemini to answer the question.

We use Gemini Flash 8B for the smaller tasks like parsing queries and determining if the current transcripts are relevant. For the larger tasks like answering questions, we use Gemini Flash.

Below we outline how the two main components work. We call them Intelligent Question Routing and Dynamic Cache Management.

## Core Strategies

### Intelligent Question Routing
OLABot employs a sophisticated approach to question analysis using LLMs. The core technical challenge is that the transcripts are quite long, in fact totalling 20 million tokens. This makes it infeasible to load the entire corpus into memory, even with Gemini's 1 million context window. We also want to optimize for resource usage. This led us to develop a system that dynamically loads and unloads transcripts based on the user's question.

First, Gemini parses every query into a set of entities:
- People (e.g., "MPP Smith", "Minister of Health")
- Bills (e.g., "Bill 124")
- Dates (e.g., "2024-01-01" -- in YYYY-MM-DD format)
- Topics (e.g., "healthcare", "education funding")

Then, Gemini determines if the current loaded transcripts are relevant to the query by comparing these entities to transcript summaries we generated, or if the current question is a follow up to the previous question. If not, we find a new set of transcripts that are relevant and load them in.


### Dynamic Cache Management
OLABot implements an innovative approach to managing conversation context:

1. **Context Relevance Analysis**
   - Uses LLM to determine if current loaded transcripts can answer new questions
   - Analyzes conversation flow and topic transitions
   - Makes intelligent decisions about when to fetch new context

2. **Adaptive Context Loading**
   - Maintains optimal context window size
   - Intelligently selects which transcripts to keep or discard
   - Balances completeness of response with performance

3. **Conversation Continuity**
   - Maintains coherent discussion across topic changes
   - Handles follow-up questions naturally
   - Preserves relevant context while discarding irrelevant information

This dual-strategy approach allows OLABot to provide more accurate and relevant responses, and scalably handle the massive transcript corpus.

## Conversation Flow

1. User submits question
2. Bot checks if current context is sufficient (`check_context_relevance`)
3. If new context needed:
   - Routes question (`route_question`)
   - Selects relevant transcripts (`select_relevant_transcripts`)
   - Updates current context (`update_current_context`)
4. Generates response using selected context (`generate_response`)
5. Updates conversation history

## Core Components

## Initialization

The bot reads a JSON file containing Hansard transcripts. It then processes them for easier searching and retrieval. It also maintains a conversation history and current transcript context, as well as two Gemini models: `model` and `small_model`, which are Gemini Flash and Gemini Flash 8B respectively, used for different tasks.

## Class Attributes
- `transcripts`: A dictionary mapping dates to transcripts
- `transcript_topics`: A dictionary mapping dates to lists of discussion topics
- `transcript_speakers`: A dictionary mapping dates to lists of speakers present
- `transcript_bills`: A dictionary mapping dates to lists of bills discussed
- `transcript_summaries`: A dictionary mapping dates to formatted summary strings
- `available_dates`: A list of all dates in the corpus in YYYY-MM-DD format, sorted in descending order
- `earliest_date`: The earliest date in the corpus
- `latest_date`: The latest date in the corpus
- `conversation_history`: Conversation history. A list of dicts, each containing a `question` and `response`
- `current_dates`: Currently loaded transcript dates. A list of strings in YYYY-MM-DD format
- `current_content`: Content of currently loaded transcripts. A dictionary mapping dates to transcripts
- `MAX_CONVERSATION_HISTORY`: Maximum number of past exchanges to maintain
- `MAX_TRANSCRIPT_CONTEXT`: Maximum number of transcripts to load at once
- `model`: Gemini Flash model
- `small_model`: Gemini Flash 8B model

## Key Methods

### Transcript Processing

```python
def generate_transcript_summaries(self) -> tuple[dict, dict, dict, dict]:
```

This function processes the transcripts into four dictionaries:
- `transcript_topics`: Maps dates to lists of discussion topics
- `transcript_speakers`: Maps dates to lists of speakers present
- `transcript_bills`: Maps dates to lists of bills discussed
- `transcript_summaries`: Maps dates to formatted summary strings

The extraction process uses heuristics with string matching, regex, and simple parsing. They are imperfect but work well for our purposes, as we will use Gemini to compare everything.

1. Topic Extraction
   - Examines the first 50 paragraphs of each transcript to find the table of contents
   - Filters out common header elements:
     - "LEGISLATIVE ASSEMBLY" (in English and French)
     - Day names (in English and French)
     - Empty lines
   - Remaining lines are considered topics for that day's session

2. Speaker Identification
   - Looks for lines such that:
     - Start with a title (Mr., Ms., Mrs., Hon., The)
     - Contains a colon
   - Then, splits the line by the colon and takes the first part as the speaker
   - Also, ensure that speaker contains either:
     - Text in parentheses, or
     - Includes a formal title
   - Example valid formats: "Mr. Smith:", "Hon. Jane Doe (Minister of Finance):", "Ms. Johnson (Toronto—Danforth):"
   - Known limitation: May occasionally miss speakers or include false positives

4. Bill Detection
   - Uses regular expressions to find bill references: `Bill\s+\d+[A-Za-z]*`
   - Example matches: "Bill 124" and "Bill 45A"

5. Summary Generation
   - Consolidated string: "DATE -- Speakers: [list] | Topics: [list] | Bills: [list]"

### Context Management

```python
def check_context_relevance(self, question: str) -> str:
```

Determines if current loaded transcripts can answer a new question by:
1. Analyzing conversation history
2. Checking current transcript summaries
3. Using Gemini to decide if new context is needed

Returns: `USE_CURRENT_CONTEXT` or `NEED_NEW_CONTEXT`

```python
def update_current_context(self, dates: list[str]):
```

Updates the current transcript context to the given dates and the corresponding transcripts.

### Question Routing

```python
def route_question(self, question: str) -> dict:
```

Analyzes questions to determine search strategy:
- Returns dictionary with:
  - `type`: TOPIC_SEARCH, PERSON_STATEMENT, or BILL_DISCUSSION
  - `topics`: List of relevant topics
  - `time_period`: Date range if specified
  - `people`: Named individuals
  - `bill_number`: Specific bill references

### Transcript Selection

```python
def select_relevant_transcripts(self, question: str) -> list[str]:
```

Selects most relevant transcript dates based on:
1. Date filters if specified
2. Speaker mentions
3. Topic relevance
4. Bill discussions

Returns: List of dates of transcripts to load in YYYY-MM-DD format

### Response Generation

```python
def generate_response(self, question: str) -> str:
```
The main method that generates responses. Uses the current question, recent conversation history, selected transcript content.

## Other Technical Details

### Error Handling
- JSON parsing errors in routing default to basic topic search
- Missing transcripts return appropriate message
- General exceptions caught in main chat loop

### Dependencies
- `google.generativeai`: Gemini AI interface
- `colorama`: Terminal output formatting
- `python-dotenv`: Environment variable management
- `json`: Transcript data handling
- `datetime`: Date processing
- `re`: Regular expression operations

### Environment Setup
Requires:
1. `.env` file with GEMINI_API_KEY
2. Hansard transcripts in JSON format
3. Python 3.x
4. Required packages installed via pip

### Debug Mode
When enabled (`debug=True`):
- Prints routing decisions
- Shows context selection process
- Displays token usage statistics

### Performance Considerations
- Maintains limited conversation history to manage context size
- Uses Gemini Flash model for routing decisions
- Limits number of loaded transcripts
- Caches current context when appropriate
