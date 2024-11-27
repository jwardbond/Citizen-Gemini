import datetime
import inspect
import json
import os
import sys
import traceback
from collections.abc import Generator
from pathlib import Path
from zoneinfo import ZoneInfo

import dotenv
import google.generativeai as genai
from colorama import Back, Fore, Style

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

TZ = ZoneInfo("America/Toronto")
"""Some Gemini helper functions"""


def list_models():
    """List all active models"""
    for m in genai.list_models():
        print(f"Model name: {m.name}")
        print(f"Display name: {m.display_name}")
        print(f"Description: {m.description}")
        print(f"Supported generation methods: {m.supported_generation_methods}")
        print("---")


def list_all_caches():
    """List all active caches"""
    for c in genai.caching.CachedContent.list():
        print(c)


def delete_all_caches():
    """Delete all active caches"""
    for c in genai.caching.CachedContent.list():
        c.delete()
        print("deleted")


"""Some string formatting and cleaning functions"""


def documents_to_string(documents: dict) -> str:
    """Takes a dictionary and combines every item into one giant string."""
    outstring = ""
    for doc_id, doc in documents.items():
        outstring += (
            f"*********************DOCUMENT {doc_id} START*********************\n"
        )
        outstring += f"ID: {doc_id}\n"
        for k, v in doc.items():
            outstring += f"{k.upper()}: {v}\n"
        outstring += (
            f"*********************DOCUMENT {doc_id} END*********************\n\n\n\n"
        )
    return outstring


def parse_llm_json(response: str) -> str:
    """Parses the JSON content from a poorly formatter LLM response.

    Sometimes, despite how much prompt torturing you do, the LLM returns, e.g.

    ```json
    {'key': val, 'key2': val2}
    ```.

    where the backticks and 'json' are literally in the string.
    This utility function will try to parse an LLM response as a json string.
    And it will also handle that annoying case where the markdown is included.
    """
    cleaned = response.strip()
    if cleaned.startswith("```"):
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]  # Remove ```json
        else:
            cleaned = cleaned[3:]  # Remove ```
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]  # Remove closing ```
        cleaned = cleaned.strip()
    return json.loads(cleaned)  # Parse to dict


class OLABot:
    """Ontario Legislature Assistant Bot.

    This class creates two gemini models and exposes a chat interface that can be used for querying
    Ontario Legislature documents, mainly the Hansard (transcripts) and bills.

    These documents can be long (sometimes 10^5 tokens or more per document). The larger of the two models
    is responsible for the actual chat functionality, but is only able to store a few documents in its
    context. The smaller of the two models is responsible for preprocessing the question, which it does by
    running it against a context cache containing *summaries* of all the documents - acting as a sort of "in
    context vector database". If the small model decides that the context of the large model is insufficient,
    it clears the context cache and specifies what new, more relevant documents should be loaded by the large
    model.

    Attributes:
        MAX_DOCUMENT_CONTEXT (int): The number of transcript files the model can store.
        CACHE_TTL_MINUTES (int): Cache time-to-live in minutes.
        RETRIEVAL_MODEL_NAME (str): The retrieval model (current set to "models/gemini-1.5-flash-8b").
        MAIN_MODEL_NAME (str): The main model (currently set to models/gemini-1.5-flash-002).
        debug (bool): Flag to enable or disable debug mode.
        streaming (bool): Flag to enable or disable streaming mode.
        documents (dict): Loaded documents data.
        summaries (dict): Loaded summaries data.
        available_dates (list): Sorted list of available transcript dates.
        available_bills (list): Sorted list of available bill numbers.
        current_document_ids (list): List of current document IDs.
        current_context (str): Current context string for the main chatbot.
        model (object): Main chatbot model.
        current_context_cache (object): Cache for the current context.
        retrieval_model (object): Context checking model.
        summaries_cache (object): Cache for the summaries.

    Args:
        documents_path (Path): Path to the documents file.
        summaries_path (Path): Path to the summaries file.
        debug (bool, optional): Flag to enable or disable debug mode. Defaults to False.
        streaming (bool, optional): Flag to enable or disable streaming mode. Defaults to True.
    """

    def __init__(
        self,
        documents_path: Path,
        summaries_path: Path,
        debug: bool = False,
        streaming: bool = True,
    ):
        """Initialize the Ontario Legislature Assistant Bot."""

        # Constants
        self.MAX_DOCUMENT_CONTEXT = 5  # The number of transcript files #FIXME this should really be max cached size or something
        self.CACHE_TTL_MINUTES = 60  # Cache time-to-live in minutes
        self.RETRIEVAL_MODEL_NAME = "models/gemini-1.5-flash-8b"
        self.MAIN_MODEL_NAME = "models/gemini-1.5-flash-002"

        # Settings
        self.debug = debug
        self.streaming = streaming

        # Load data
        if isinstance(documents_path, str):
            documents_path = Path(documents_path)
        with documents_path.open(encoding="utf-8") as f:
            self.documents = json.load(f)

        if isinstance(summaries_path, str):
            summaries_path = Path(summaries_path)
        with summaries_path.open(encoding="utf-8") as f:
            self.summaries = json.load(f)

        # Get the date range of available transcripts
        available_dates = [
            v["id_number"] for v in self.documents.values() if v["type"] == "transcript"
        ]
        self.available_dates = sorted(available_dates)

        # Get the bills numbers for the available documents
        available_bills = [
            v["id_number"] for v in self.documents.values() if v["type"] == "bill"
        ]
        self.available_bills = sorted(available_bills)

        # Create context for the main chatbot
        # initilized to the three most recent transcripts
        self.current_document_ids = [
            f"transcript {d}" for d in self.available_dates[-3:]
        ]
        self.current_context = documents_to_string(
            {k: self.documents[k] for k in self.current_document_ids},
        )

        # Initialize main chatbot
        self._print_debug(
            f"Initializing model with {', '.join(self.current_document_ids)}",
        )
        self.model, self.current_context_cache = self._create_cached_model(
            model_name=self.MAIN_MODEL_NAME,
            display_name="OLABot",
            contents=[self.current_context],
            temperature=1,
            max_output_tokens=8192,
        )

        # Initialize context checking model
        self.retrieval_model, self.summaries_cache = self._create_cached_model(
            model_name=self.RETRIEVAL_MODEL_NAME,
            display_name="OLABot Retrieval Helper",
            contents=[documents_to_string(self.summaries)],
            temperature=1,
            max_output_tokens=1024,
        )

        # Create the initial chat session
        # this is initialized to none here as a placeholder, and gets
        # set/reset whenever using the main model whenever
        # _initialize_chat_session is called
        self.chat_session = None
        self._initialize_chat_session()

    #
    # PRINTING METHODS
    #
    def _print_debug(self, message: str) -> None:
        """Print debug messages in dim yellow w. timestamp."""
        time = datetime.datetime.now(tz=TZ).time()
        if self.debug:
            print(
                f"\n{Fore.YELLOW}{Style.DIM}Debug [{time}]\n{message}{Style.RESET_ALL}"
            )

    def _format_usage_stats(self, usage_stats: dict) -> None:
        """Prints formatted usage statistics for a Gemini API call using _print_debug."""
        operation_name = (
            f"{Fore.RED}{Style.DIM}{inspect.stack()[1][3]}{Fore.YELLOW}{Style.DIM}"
        )

        stats_message = (
            f"{operation_name} usage stats:\n"
            f"\tprompt_token_count: {usage_stats.prompt_token_count}\n"
            f"\tcandidates_token_count: {usage_stats.candidates_token_count}\n"
            f"\ttotal_token_count: {usage_stats.total_token_count}"
        )

        return stats_message

    def print_welcome(self) -> None:
        """Prints welcome message."""
        print(f"\n{Back.BLUE}{Fore.WHITE} CITIZEN GEMINI {Style.RESET_ALL}")
        print(f"{Fore.CYAN}Your AI Guide to the Ontario Legislature{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Type 'quit' to exit{Style.RESET_ALL}\n")

    def print_question(self, question: str) -> None:
        """Prints user question with styling."""
        print(f"\n{Fore.GREEN}❓ You: {Style.BRIGHT}{question}{Style.RESET_ALL}")

    def print_response(self, response: str | Generator) -> None:
        """Prints bot response with styling."""
        if self.streaming:
            # Print assistant prefix only once
            print(f"\n{Fore.BLUE}🤖 OLABot: {Style.NORMAL}", end="", flush=True)
            for chunk in response:
                print(f"{Fore.BLUE}{chunk}", end="", flush=True)
            print(f"\n\n{Style.DIM}---{Style.RESET_ALL}")  # Separator line
        else:
            print(
                f"\n{Fore.BLUE}🤖 OLABot: {Style.NORMAL}{response}{Style.RESET_ALL}\n"
            )
            print(f"{Style.DIM}---{Style.RESET_ALL}")  # Separator line

    #
    # GEMINI INIT METHODS
    #
    def _create_cached_model(
        self,
        model_name: str,
        display_name: str,
        contents: list[str],
        **kwargs,
    ) -> tuple[genai.GenerativeModel, genai.caching.CachedContent | None]:
        """Create a Gemini model with cached content.

        Args:
            model_name (str): Name of the Gemini model to use
            display_name (str): Display name for the cached content
            contents (List[str]): List of content strings to cache
            **kwargs: Additional configuration parameters for the model

        Returns:
            Tuple[GenerativeModel, CachedContent]
        """

        # Create new cache
        cached_content = genai.caching.CachedContent.create(
            model=model_name,
            display_name=display_name,
            contents=contents,
            ttl=datetime.timedelta(minutes=self.CACHE_TTL_MINUTES),
        )

        # Create model with provided config
        model = genai.GenerativeModel.from_cached_content(
            cached_content=cached_content,
            generation_config={"response_mime_type": "text/plain", **kwargs},
        )
        self._print_debug(f"Created cached model: {display_name}")

        return model, cached_content

    def _initialize_chat_session(
        self,
        previous_history: list[dict] | None = None,
    ):
        """Initialize chat with system prompt and optional previous history.

        Resets self.chat_session.
        """

        system_prompt = """
        You are a helpful assistant making the Ontario Legislative Assembly more accessible to average citizens.

        IMPORTANT: You have persistent access to cached Ontario Legislative Assembly documents, including transcripts and bills.
        These documents remain available to you throughout the conversation and you should use them fully.

        HOW TO ANSWER QUESTIONS:

        1. CONTENT AND STYLE
        - Use plain, accessible language - avoid political jargon
        - Break down complex legislative concepts
        - Structure responses in short, clear paragraphs
        - Include specific dates when referencing discussions
        - Be factual and neutral in tone

        2. USING CACHED DOCUMENTS
        - Reference and quote directly from cached documents
        - Cite specific dates and sessions when quoting
        - Provide context for any technical terms or procedures
        - Connect legislative discussions to real-world impacts

        3. LEVEL OF DETAIL
        - Start with a concise summary
        - Add relevant details based on the question's scope
        - If asked for more detail, provide comprehensive information
        - Include specific examples from the documents when helpful

        4. CLARITY AND COMPLETENESS
        - If something is unclear in the documents, say so
        - If multiple documents are relevant, synthesize the information
        - Explain legislative procedures in citizen-friendly terms

        Remember: Your audience is the average citizen who wants to understand their government's activities. Make complex
        legislative information accessible while making full use of your cached documents.
        """

        initial_history = [
            {"role": "user", "parts": system_prompt},
            {"role": "model", "parts": "I understand my role and guidelines."},
        ]

        if previous_history:
            initial_history.extend(previous_history)

        if not previous_history:
            self._print_debug("Initialized chat with system prompt")
        else:
            self._print_debug("Carried over previous history")

        self.chat_session = self.model.start_chat(history=initial_history)

    #
    # CONTEXT METHODS
    #
    def _update_current_context(self, document_ids: list[str]) -> None:
        """Update the current chat context."""
        self.current_document_ids = document_ids[: self.MAX_DOCUMENT_CONTEXT]
        self.current_context = documents_to_string(
            {k: self.documents[k] for k in self.current_document_ids}
        )

        # Save previous history
        previous_history = (
            self.chat_session.history[2:] if hasattr(self, "chat_session") else None
        )

        # Delete old cache if it exists
        if hasattr(self, "current_context_cache"):
            self.current_context_cache.delete()
            self._print_debug("Deleted old cache")

        # Create new model using helper function
        self.model, self.current_context_cache = self._create_cached_model(
            model_name=self.MAIN_MODEL_NAME,
            display_name="OLABot",
            contents=[self.current_context],
            temperature=1,
            max_output_tokens=8192,
        )

        # Initialize chat with new context and previous history
        self._initialize_chat_session(previous_history)

    def _check_context_relevance(self, question: str) -> bool:
        """Check if the current context can answer the new question."""

        # No context to answer the question
        if self.current_document_ids == []:
            return False

        # Get conversation history from chat
        conversation_history = (
            "\n".join(
                [
                    f"Previous {msg.role}: {msg.parts[0].text}"
                    for msg in self.chat_session.history[2:]
                ]  # Skip system prompt
            )
            if self.chat_session.history
            else ""
        )

        prompt = f"""
        QUESTION: "{question}"

        CONTEXT INFORMATION:
        1. You have access to summaries of ALL documents in the format:
        ***DOCUMENT {{type}} {{id}} START***
        SUMMARY: [content summary]
        ***DOCUMENT {{type}} {{id}} END***

        2. Currently loaded COMPLETE documents: {', '.join(self.current_document_ids)}
        (These are the only documents available for detailed analysis)

        3. CONVERSATION FLOW in history:
        - Previous questions from the user
        - The main assistant's answers to those questions
        - Most recent exchange used these loaded documents

        EVALUATION STEPS:
        1. READ the main assistant's last answer carefully - what information was already provided?
        2. Review the conversation flow:
        a) What did the user previously ask?
        b) What specific information did the main assistant provide?
        c) Does the answer contain ANY information about the new question's topic?
        3. For the new question, check:
        a) Does it ask for more details about something already mentioned?
        b) Does it use words like "this" referring to previous content?
        c) Is it asking to expand on a point already touched upon?
        4. Important: If the previous answer contained ANY relevant information about the topic,
        USE_CURRENT_CONTEXT to expand on that information before loading new context.

        DECISION RULES:

        USE_CURRENT_CONTEXT when ANY of these are true:
        1. The main assistant's last answer contains ANY information about the topic being asked about
        2. The question asks for more details/examples/explanation of something mentioned in the last answer
        3. The question uses words like "this", "these", "those" referring to content from the last answer
        4. The question is about reactions/responses/criticism/support related to what was just discussed
        5. The question explicitly asks about current context

        LOAD_NEW_CONTEXT only when ALL of these are true:
        1. The main assistant's last answer contains NO information about the topic being asked about
        2. The question asks about documents that aren't currently loaded
        3. The question requires searching across a broader set of documents
        4. The question introduces a completely new topic unrelated to the last answer
        5. The question can't possibly be answered using information from current documents

        EXAMPLES:

        Example 1:
        Previous Question: "What is Bill 118 about?"
        Previous Answer: "Bill 118 establishes June 1st as Injured Workers Day..."
        New Question: "Can you explain more about the workplace provisions in this bill?"
        Current Documents: ["bill 118", "transcript 2023-10-23"]
        Decision: USE_CURRENT_CONTEXT
        Reasoning: Follows directly from previous Q&A about Bill 118's contents

        Example 2:
        Previous Question: "What does Bill 212 say about transportation?"
        Previous Answer: "Bill 212 includes provisions about bicycle lanes that allow the province to override municipal decisions..."
        New Question: "I'm concerned about the bicycle lane changes, can you provide more details?"
        Current Documents: ["bill 212", "transcript 2024-11-21"]
        Decision: USE_CURRENT_CONTEXT
        Reasoning: Directly references information provided in last answer about bicycle lanes

        Example 3:
        Previous Question: "What did MPPs say about healthcare funding?"
        Previous Answer: "During the May debates, MPPs discussed hospital funding..."
        New Question: "What does Bill 124 say about nurses?"
        Current Documents: ["transcript 2024-05-30"]
        Decision: LOAD_NEW_CONTEXT
        Reasoning: Switches from transcript discussions to requesting specific bill content not currently loaded

        Example 4:
        Previous Question: "What did the Premier say about healthcare?"
        Previous Answer: "In the Question Period, Premier Ford highlighted investments including 12,500 licensed physicians and 100% match in residency positions..."
        New Question: "Can you tell me more about the residency position expansion?"
        Current Documents: ["transcript 2024-05-08", "transcript 2024-03-20"]
        Decision: USE_CURRENT_CONTEXT
        Reasoning: Asks for elaboration on specific detail mentioned in previous answer

        Example 5:
        Previous Question: "What were the key points in Bill 75?"
        Previous Answer: "Bill 75 focuses on housing development regulations..."
        New Question: "Are there any other bills about housing from this session?"
        Current Documents: ["bill 75", "transcript 2024-03-15"]
        Decision: LOAD_NEW_CONTEXT
        Reasoning: Requires broader search across multiple bills beyond current context

        Example 6:
        Previous Question: "What bills are currently loaded?"
        Previous Answer: "The currently loaded documents include Bill 124 and several transcripts..."
        New Question: "What other bills mention healthcare?"
        Current Documents: ["bill 124", "transcript 2024-05-30"]
        Decision: LOAD_NEW_CONTEXT
        Reasoning: Despite being related to current topic, explicitly requests other bills

        Example 7:
        Previous Question: "Tell me about the housing crisis discussion"
        Previous Answer: "In these October sessions, MPPs debated housing affordability..."
        New Question: "What else was said in these sessions?"
        Current Documents: ["transcript 2023-10-23", "transcript 2023-10-24"]
        Decision: USE_CURRENT_CONTEXT
        Reasoning: Explicitly refers to "these sessions" in currently loaded transcripts

        Example 8:
        Previous Question: "What happened in the November 15th session?"
        Previous Answer: "The November 15th session covered several topics including education funding..."
        New Question: "Were there any bills introduced that day?"
        Current Documents: ["transcript 2023-11-15"]
        Decision: USE_CURRENT_CONTEXT
        Reasoning: Asks about same session already loaded, just different aspect

        Example 9:
        Previous Question: "What did Minister Jones say about healthcare?"
        Previous Answer: "Minister Jones discussed hospital funding and staffing initiatives..."
        New Question: "Did any opposition members respond to these points?"
        Current Documents: ["transcript 2024-05-30", "transcript 2024-05-31"]
        Decision: USE_CURRENT_CONTEXT
        Reasoning: Asks about responses to specific points in current transcripts

        Example 10:
        Previous Question: "What does the current context contain?"
        Previous Answer: "Currently loaded documents include Bill 118 and transcript from October 23..."
        New Question: "Great, can you search other transcripts for mentions of Bill 118?"
        Current Documents: ["bill 118", "transcript 2023-10-23"]
        Decision: LOAD_NEW_CONTEXT
        Reasoning: Explicitly requests search beyond current documents

        Example 11:
        Previous Question: "What's in Bill 124?"
        Previous Answer: "Bill 124 deals with healthcare worker compensation..."
        New Question: "Show me what's currently loaded"
        Current Documents: ["bill 124", "transcript 2024-05-30"]
        Decision: USE_CURRENT_CONTEXT
        Reasoning: Explicitly asks about current context contents

        Example 12:
        Previous Question: "Did they discuss climate change in October?"
        Previous Answer: "Yes, in the October 23rd session, MPPs debated environmental policies..."
        New Question: "What was said about this in other months?"
        Current Documents: ["transcript 2023-10-23"]
        Decision: LOAD_NEW_CONTEXT
        Reasoning: "Other months" explicitly requests searching beyond current transcripts

        RESPONSE FORMAT:
        {{
            "decision": "USE_CURRENT_CONTEXT" or "LOAD_NEW_CONTEXT",
            "reasoning": "<brief explanation focusing on relationship to previous Q&A exchange>"
        }}
        Do NOT include any markdown formatting or backticks in the final answer. Just return a JSON that can be parsed.
        """

        response = self.retrieval_model.generate_content(prompt)
        response_dct = parse_llm_json(response.text)
        decision = response_dct["decision"]
        reasoning = response_dct["reasoning"]
        self._print_debug(self.current_document_ids)
        self._print_debug(self._format_usage_stats(response.usage_metadata))
        self._print_debug(f"Checking context relevance decision: {decision}")
        self._print_debug(f"Checking context relevance reasoning: {reasoning}")
        return decision == "USE_CURRENT_CONTEXT"

    #
    # FILTERING METHODS
    #
    def _select_relevant_documents(self, question: str) -> list[str]:
        """Select relevant transcripts based on question and available summaries."""

        prompt = f"""
        Given the following question about the Ontario Legislature, and using the document summaries provided in your context,
        select the most relevant documents to help you answer the question.

        QUESTION: "{question}"

        The documents in your context are TRANSCRIPTS and BILLS from the current Ontario Legislative Assembly.
        The transcript dates range from {self.available_dates[0]} to {self.available_dates[-1]} and the bill numbers ranging from numbers {self.available_bills[0]} to {self.available_bills[-1]}

        CRITICAL PRIORITY: Always prefer answering general questions with more recent transcripts. For example: "What has Doug Ford said about healthcare?" should be answered preferring more recent information.

        CRITICAL PRIORITY: If the question is explicitly asking for more information on a bill, or about a specific part of a bill, you should select that bill.

        CRITICAL PRIORITY: If the question is around the DISCUSSION of a bill, you can prioritize transcripts over the bill itself.

        Consider in order of priority:
        1. Most recent documents (starting from {self.available_dates[-1]})
        1. Explicit date references in the question
        2. Explicit bill references in the question
        3. Speakers mentioned
        4. Topics and their semantic similarities
        5. Bill discussions

        Transcripts have ids like "transcript YYYY-MM-DD"
        Bills have ids like "bill 123"

        Return ONLY the document ids, one on each line, without asterisks.
        Limit your response to {self.MAX_DOCUMENT_CONTEXT} documents maximum.
        Start with the highest priority document
        """

        # get most recent relevant
        response = self.retrieval_model.generate_content(prompt)
        response_text = response.text.replace("*", "")  # replace asterisks just in case
        relevant_documents = [
            doc for doc in response_text.split("\n") if doc.strip() in self.documents
        ]

        # fallback: if none returned
        if not relevant_documents:
            self._print_debug("No specific dates found, using most recent transcripts")
            relevant_documents = self.available_dates[
                -(self.MAX_DOCUMENT_CONTEXT + 1) :
            ]

        # Clip to max context length
        relevant_documents = relevant_documents[
            : self.MAX_DOCUMENT_CONTEXT
        ]  # TODO better clipping

        self._print_debug(self._format_usage_stats(response.usage_metadata))
        self._print_debug(f"Selected documents: {relevant_documents}")
        return relevant_documents

    #
    # RESPONSE GENERATION METHODS
    #
    def _generate_response(self, question: str) -> str | Generator:
        """Generate response using selected transcripts.

        Returns a string if not streaming, otherwise a generator.
        """

        try:
            response = self.chat_session.send_message(question, stream=self.streaming)
            self._print_debug("Generated response")  # Add debug logging

            if not self.streaming:
                response_text = response.text
                return response_text

            return response

        except Exception as e:
            self._print_debug(f"Error generating response: {str(e)}")
            raise

    def chat_interface(self, question: str) -> Generator:
        """Main chat interface."""
        try:
            # Check if current context is relevant
            if self._check_context_relevance(question):
                self._print_debug("Using cached context")
                response = self._generate_response(question)
            # Load new context
            else:
                self._print_debug("Fetching new context")
                relevant_documents = self._select_relevant_documents(question)
                self._update_current_context(relevant_documents)

                if not relevant_documents:
                    return "I couldn't find any relevant discussions in the available transcripts."

                response = self._generate_response(question)

            if not self.streaming:
                yield response

            # handle streaming
            full_response = ""
            for chunk in response:
                full_response += chunk.text
                yield chunk.text

            self._print_debug(self._format_usage_stats(chunk.usage_metadata))

        # catchall error handling for the whole chatbot
        # so try to print as much detail as possible
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self._print_debug(
                "".join(traceback.format_exception_only(exc_type, exc_value)).strip()
            )
            self._print_debug("".join(traceback.format_tb(exc_traceback)))

            error_msg = (
                f"{Fore.RED}Sorry, I encountered an error: {e!s}{Style.RESET_ALL}"
            )
            yield error_msg


def main():
    # Clear all caches in the cloud
    delete_all_caches()

    # Initialize bot
    bot = OLABot("./documents.json", "./summaries.json", debug=True)
    bot.print_welcome()

    try:
        while True:
            question = input(f"{Fore.GREEN}Your question: {Style.RESET_ALL}")
            if question.lower() == "quit":
                break

            bot.print_question(question)
            bot._print_debug(datetime.datetime.now(tz=TZ).time())
            if not bot.streaming:
                response = next(bot.chat_interface(question))
            if bot.streaming:
                response = bot.chat_interface(question)
            bot.print_response(response)
    except KeyboardInterrupt:
        bot._print_debug("Interrupted by user.")
    finally:
        try:
            bot.current_context_cache.delete()
            bot.summaries_cache.delete()
            bot._print_debug("Deleted all caches")
        except Exception as e:
            bot._print_debug(f"Error cleaning up caches: {e}")

        print(f"\n{Fore.CYAN}Goodbye! 👋{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
