import datetime
import inspect
import json
import os
from collections.abc import Generator
from pathlib import Path
from zoneinfo import ZoneInfo

import dotenv
import google.generativeai as genai
from colorama import Back, Fore, Style
from google.generativeai import caching

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

TZ = ZoneInfo("America/Toronto")
"""Some Gemini helper functions"""


def list_models():
    for m in genai.list_models():
        print(f"Model name: {m.name}")
        print(f"Display name: {m.display_name}")
        print(f"Description: {m.description}")
        print(f"Supported generation methods: {m.supported_generation_methods}")
        print("---")


def list_all_caches():
    for c in caching.CachedContent.list():
        print(c)


def delete_all_caches():
    for c in caching.CachedContent.list():
        c.delete()
        print("deleted")


def documents_to_string(documents: dict) -> str:
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


class OLABot:
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
        self.CACHE_TTL_MINUTES = 10  # Cache time-to-live in minutes
        self.RETRIEVAL_MODEL_NAME = "models/gemini-1.5-flash-8b"
        self.MAIN_MODEL_NAME = "models/gemini-1.5-flash-002"

        # Settings
        self.debug = debug
        self.streaming = streaming

        # Load data
        documents_path = Path(documents_path)
        with documents_path.open(encoding="utf-8") as f:
            self.documents = json.load(f)
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
            v["id_number"] for v in self.documents.values() if v["type"] == "bills"
        ]
        self.available_bills = sorted(available_bills)

        # Create context for the main chatbot
        # initilized to the three most recent transcripts
        # **ID**\n Type: [type] \n\n [text]
        self.current_document_ids = [
            f"transcript {d}" for d in self.available_dates[-3:]
        ]
        self.current_context = documents_to_string(
            {k: self.documents[k] for k in self.current_document_ids}
        )

        # Initialize main chatbot
        self._print_debug(
            f"Initializing model with {", ".join(self.current_document_ids)}",
        )
        self.model, self.current_context_cache = self._create_cached_model(
            model_name=self.MAIN_MODEL_NAME,
            display_name="OLABot Current Transcripts",
            contents=[self.current_context],
            temperature=1,
            max_output_tokens=8192,
        )

        # Initialize context checking model #TODO what should we call this model
        self.retrieval_model, self.summaries_cache = self._create_cached_model(
            model_name=self.RETRIEVAL_MODEL_NAME,
            display_name="OLABot Transcript Summaries",
            contents=[documents_to_string(self.summaries)],
            temperature=1,  # TODO check temp
            max_output_tokens=1024,
        )

        # Create the initial chat session
        # this is initialized to none here as a placeholder, and gets
        # set/reset whenever using the main model whenever
        # _initialize_chat_session is called
        self.chat = None  # TODO take out?
        self._initialize_chat_session()

    #
    # PRINTING METHODS
    #
    def _print_debug(self, message: str) -> None:
        """Print debug messages in dim yellow."""
        time = datetime.datetime.now(tz=TZ).time()
        if self.debug:
            print(
                f"\n{Fore.YELLOW}{Style.DIM}Debug [{time}]\n{message}{Style.RESET_ALL}"
            )

    def _format_usage_stats(self, usage_stats: dict) -> None:
        """Prints formatted usage statistics for a Gemini API call using _print_debug.

        Args:
            usage_stats: The usage_metadata from a Gemini response
        """
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
        """Print welcome message."""
        print(f"\n{Back.BLUE}{Fore.WHITE} CITIZEN GEMINI {Style.RESET_ALL}")
        print(f"{Fore.CYAN}Your AI Guide to the Ontario Legislature{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Type 'quit' to exit{Style.RESET_ALL}\n")

    def print_question(self, question: str) -> None:
        """Print user question with styling."""
        print(f"\n{Fore.GREEN}❓ You: {Style.BRIGHT}{question}{Style.RESET_ALL}")

    def print_response(self, response: str | Generator) -> None:
        """Print bot response with styling."""
        if self.streaming:
            # Print assistant prefix only once
            print(f"\n{Fore.BLUE}🤖 Assistant: {Style.NORMAL}", end="", flush=True)
            for chunk in response:
                print(f"{Fore.BLUE}{chunk}", end="", flush=True)
            print(f"\n\n{Style.DIM}---{Style.RESET_ALL}")  # Separator line
        else:
            print(
                f"\n{Fore.BLUE}🤖 Assistant: {Style.NORMAL}{response}{Style.RESET_ALL}\n"
            )
            print(f"{Style.DIM}---{Style.RESET_ALL}")  # Separator line

    #
    # GEMINI METHODS
    #
    def _create_cached_model(
        self,
        model_name: str,
        display_name: str,
        contents: list[str],
        **kwargs,
    ) -> tuple[genai.GenerativeModel, caching.CachedContent | None]:
        """Create a Gemini model with cached content.

        Args:
            model_name: Name of the Gemini model to use
            display_name: Display name for the cached content
            contents: List of content strings to cache
            **kwargs: Additional configuration parameters for the model

        Returns:
            Tuple of (GenerativeModel, CachedContent)
        """

        # Create new cache
        cached_content = caching.CachedContent.create(
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

        Returns:
            Gemini chat session object
        """

        system_prompt = """You are a helpful assistant making the Ontario Legislative Assembly more accessible to average citizens.
        You are given a question and a set DOCUMENTS which contains both TRANSCRIPTS and/or BILLS from the current Ontario Legislative Assembly.
        Be specific and concise in your responses. Do not talk about irrelevant things in the documents.
        Make sure to use all the information available to you to answer the question.

        Guidelines for your response:
        1. Be concise and clear - avoid political jargon
        2. If this is a follow-up question, reference relevant information from previous responses
        3. If quoting from documents, only use the most relevant quotes
        4. Structure your response in short paragraphs
        5. Include the date when mentioning specific discussions
        6. If something is unclear or missing from the documents, say so

        Remember: Your audience is the average citizen who wants to understand what is going on in their government legislature.
        If the audience asks for more detail, feel free to provide a more comprehensive answer."""

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
            display_name="OLABot Current Transcripts",
            contents=[self.current_context],
            temperature=1,
            max_output_tokens=8192,
        )

        # Initialize chat with new context and previous history
        self._initialize_chat_session(previous_history)

    def _check_context_relevance(self, question: str) -> bool:
        """Check if the current context can answer the new question."""

        # No context to answer the question
        if self.current_context == "":
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
        New question:
        "{question}"

        Previous conversation:
        {conversation_history}

        Your task: Determine if the question is answerable by a different model, which only has access to the following documents: {', '.join(self.current_document_ids)} 
        and the previous conversation history provided above. You can tell this second model to USE_CURRENT_CONTEXT if it should be able to answer the question 
        with those documents or you can tell it to LOAD_NEW_CONTEXT.

        Return USE_CURRENT_CONTEXT if ANY of these are true:
        - Question can likely be completely answered with its current documents
        - Question is a follow-up or clarification of it's previous discussion
        - Question uses words like "that", "this", "those", "they" referring to the documents already in its context
        - Question asks for more details about topics/people already discussed
        - Question is about interpretation or analysis of documents already in its context
        - Question is about a bill and refers to documents already in its context

        Return LOAD_NEW_CONTEXT if ANY of these are true:
        - Question explicitly asks for information on dates/meetings OR bills that aren't in its current context
        - Question mentions specific people/topics definitely not in its current context
        - Question contains explicit search requests like "find all instances" or "search all transcripts"
        - Current context is completely irrelevant to the new question
        - You are 100% certain it will require new transcripts
        - You are 100% certain it will require new bills

        Return either: LOAD_NEW_CONTEXT or USE_CURRENT_CONTEXT
        """  # TODO change current context to currently loaded documents

        response = self.retrieval_model.generate_content(prompt)
        decision = response.text.strip()
        self._print_debug(self.current_document_ids)
        self._print_debug(self._format_usage_stats(response.usage_metadata))
        self._print_debug(f"Checking context relevance decision: {decision}")
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

    def chat_interface(self, question: str) -> str | Generator:
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
                return response

            # handle streaming
            full_response = ""
            for chunk in response:
                full_response += chunk.text
                yield chunk.text

            self._print_debug(self._format_usage_stats(chunk.usage_metadata))

        except Exception as e:
            return f"{Fore.RED}Sorry, I encountered an error: {e!s}{Style.RESET_ALL}"


def main():
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
