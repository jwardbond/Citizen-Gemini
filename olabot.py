import inspect
import json
import os
import sys
import re
from pathlib import Path
from typing import Union, Generator
import dotenv
import google.generativeai as genai
from google.generativeai import caching
from colorama import Back, Fore, Style
import datetime

dotenv.load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


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


class OLABot:
    def __init__(self, hansard_path: Path, debug: bool = False, streaming: bool = True):
        """Initialize the Ontario Legislature Assistant Bot."""
        # some constants
        self.MAX_TRANSCRIPT_CONTEXT = 5  # The number of transcript files
        self.CACHE_TTL_MINUTES = 10  # Cache time-to-live in minutes
        self.RETRIEVAL_MODEL_NAME = 'models/gemini-1.5-flash-8b'
        self.MAIN_MODEL_NAME = 'models/gemini-1.5-flash-002'

        # some settings
        self.debug = debug
        self.streaming = streaming

        # Load transcripts
        if not isinstance(hansard_path, Path):
            hansard_path = Path(hansard_path)
        with hansard_path.open(encoding="utf-8") as f:
            self.transcripts = json.load(f)
        # Get the date range of available transcripts
        self.available_dates = sorted(self.transcripts.keys(), reverse=True)
        self.earliest_date = self.available_dates[-1]
        self.latest_date = self.available_dates[0]

        (
            self.transcript_speakers,
            self.transcript_topics,
            self.transcript_bills,
            self.transcript_summaries,
        ) = self._generate_transcript_summaries()
        self.transcript_condensed = "\n\n".join(
            [f"**{date}**:\n{self.transcript_summaries[date]}" for date in self.available_dates]
        )

        # Initialize models
        self.model, self.current_context_cache = self._create_cached_model(
            model_name=self.MAIN_MODEL_NAME,
            display_name='OLABot Current Transcripts',
            contents=[''],
            temperature=1,
            max_output_tokens=8192
        )
        self._initialize_chat() # assigns

        self.retrieval_model, self.transcript_summaries_cache = self._create_cached_model(
            model_name=self.RETRIEVAL_MODEL_NAME,
            display_name='OLABot Transcript Summaries',
            contents=[self.transcript_condensed],
            temperature=1,
            max_output_tokens=1024
        )

        # Initialize current transcript context
        self.current_dates = []
        self.current_context = ""

    #
    # TRANSCRIPT SUMMARIES
    #
    def _generate_transcript_summaries(self) -> tuple:
        """Process and store topics and speakers for each transcript."""
        transcript_topics = {}
        transcript_speakers = {}
        transcript_bills = {}
        for date, transcript in self.transcripts.items():
            toc_lines = transcript.split("\n\n")[0:50]
            # Skip header lines and empty lines
            topics = [
                line.strip()
                for line in toc_lines
                if line.strip()
                and "LEGISLATIVE ASSEMBLY" not in line
                and "ASSEMBLÉE LÉGISLATIVE" not in line
                and not line.startswith("Thursday")
                and not line.startswith("Monday")
                and not line.startswith("Tuesday")
                and not line.startswith("Wednesday")
                and not line.startswith("Friday")
                and not line.startswith("Jeudi")
                and not line.startswith("Lundi")
                and not line.startswith("Mardi")
                and not line.startswith("Mercredi")
                and not line.startswith("Vendredi")
            ]
            transcript_topics[date] = topics

            # Find speakers in Question Period and other sections
            # Look for patterns like:
            # "Mr. Smith:", "Ms. Jones:", "Hon. Doug Ford:", "The Speaker:"
            # This check is imperfect, but it's a good enough heuristic
            # And we will toss it into LLM anyways
            """
            Example of failure mode (has colon and starts with The):
            The 911 model of care that we referenced at the Association of Municipalities of Ontario conference earlier this week has been embraced: community paramedicine that allows community paramedics to go into those homes, for individuals who are able, in most cases with very little support, to stay safely in their home. The municipalities that have embraced that 911 model of care have loved it. In fact, our satisfaction rate, I believe, is in the 97th percentile.
            """
            speakers = set()
            lines = transcript.split("\n")
            for line in lines:
                # Process lines that start with a title prefix
                if (
                    any(
                        line.strip().startswith(prefix)
                        for prefix in ["Mr.", "Ms.", "Mrs.", "Hon.", "The"]
                    )
                    and ":" in line
                ):
                    speaker = line.split(":")[0].strip()
                    if ("(" in speaker and ")" in speaker) or any(
                        title in speaker for title in ["Mr.", "Ms.", "Mrs.", "Hon."]
                    ):
                        speakers.add(speaker)

            transcript_speakers[date] = list(speakers)

            # Find bills
            # Bills have a simple pattern: "Bill 123A"
            bills = set()
            bill_pattern = re.compile(r"Bill\s+\d+[A-Za-z]*")

            for line in lines:
                bill_matches = bill_pattern.findall(line)
                bills.update(bill_matches)

            transcript_bills[date] = list(bills)

        transcript_summaries = {
            date: f"{date} -- Speakers: {', '.join(transcript_speakers[date])} | Topics: {', '.join(transcript_topics[date])} | Bills: {', '.join(transcript_bills[date]) if date in transcript_bills else 'None'}"
            for date in self.available_dates
        }

        return (
            transcript_topics,
            transcript_speakers,
            transcript_bills,
            transcript_summaries,
        )

    #
    # PRINTING METHODS
    #
    def _print_debug(self, message: str) -> None:
        """Print debug messages in dim yellow."""
        if self.debug:
            print(f"{Fore.YELLOW}{Style.DIM}Debug - {message}{Style.RESET_ALL}")

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
            print(f"\n{Fore.BLUE}🤖 Assistant: {Style.NORMAL}{response}{Style.RESET_ALL}\n")
            print(f"{Style.DIM}---{Style.RESET_ALL}")  # Separator line

    #
    # GEMINI METHODS
    #
    def _create_cached_model(
        self,
        model_name: str,
        display_name: str,
        contents: list[str],
        **kwargs
    ) -> tuple[genai.GenerativeModel, caching.CachedContent | None]:
        """Create a Gemini model with cached content.

        Args:
            model_name: Name of the Gemini model to use
            display_name: Display name for the cached content
            contents: List of content strings to cache
            **kwargs: Additional configuration parameters for the model

        Returns:
            Tuple of (GenerativeModel, CachedContent or None if fallback)
        """
        try:
            # Create new cache
            cached_content = caching.CachedContent.create(
                model=model_name,
                display_name=display_name,
                contents=contents,
                ttl=datetime.timedelta(minutes=self.CACHE_TTL_MINUTES)
            )

            # Create model with provided config
            model = genai.GenerativeModel.from_cached_content(
                cached_content=cached_content,
                generation_config={
                    "response_mime_type": "text/plain",
                    **kwargs
                }
            )
            self._print_debug(f"Created cached model: {display_name}")
            return model, cached_content

        except Exception as e:
            self._print_debug(f"Error creating cached model: {str(e)}")
            # Fallback to non-cached model
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "response_mime_type": "text/plain",
                    **kwargs
                }
            )
            self._print_debug(f"Fell back to non-cached model: {display_name}")
            return model, None

    def _initialize_chat(self, previous_history: list[dict] = None) -> None:
        """Initialize chat with system prompt and optional previous history."""

        system_prompt = """You are a helpful assistant making the Ontario Legislature more accessible to average citizens.
        You are given a question and a set of transcripts from the Ontario Legislature.
        Be specific and concise in your responses. Do not talk about irrelevant things in the transcripts.
        Make sure to use all the information available to you to answer the question.

        Guidelines for your response:
        1. Be concise and clear - avoid political jargon
        2. If this is a follow-up question, reference relevant information from previous responses
        3. If quoting from transcripts, only use the most relevant quotes
        4. Structure your response in short paragraphs
        5. Include the date when mentioning specific discussions
        6. If something is unclear or missing from the transcripts, say so

        Remember: Your audience is the average citizen who wants to understand what happened in their legislature.
        If the audience asks for more detail, feel free to provide a more comprehensive answer."""

        initial_history = [
            {"role": "user", "parts": system_prompt},
            {"role": "model", "parts": "I understand my role and guidelines."}
        ]

        if previous_history:
            initial_history.extend(previous_history)

        self.chat = self.model.start_chat(history=initial_history)

        if not previous_history:
            self._print_debug("Initialized chat with system prompt")
        else:
            self._print_debug("Carried over previous history")

    #
    # CONTEXT METHODS
    #
    def _update_current_context(self, dates: list[str]) -> None:
        """Update the current transcript context."""
        self.current_dates = dates
        self.current_context = "\n\n".join(
            [
                f"**Transcript from {date}**:\n{self.transcripts[date]}"
                for date in dates[: self.MAX_TRANSCRIPT_CONTEXT]
            ],
        )

        try:
            # Save previous history
            previous_history = self.chat.history[2:] if hasattr(self, 'chat') else None

            # Delete old cache if it exists
            if hasattr(self, 'current_context_cache'):
                self.current_context_cache.delete()
                self._print_debug("Deleted old cache")

            # Create new model using helper function
            self.model, self.current_context_cache = self._create_cached_model(
                model_name=self.MAIN_MODEL_NAME,
                display_name='OLABot Current Transcripts',
                contents=[self.current_context],
                temperature=1,
                max_output_tokens=8192
            )

            # Initialize chat with new context and previous history
            self._initialize_chat(previous_history)
            self._print_debug("Updated main model with new cached context")
        except Exception as e:
            self._print_debug(f"Error updating context cache: {str(e)}")
            # Fallback to non-cached model
            self.model = genai.GenerativeModel(
                model_name=self.MAIN_MODEL_NAME,
                generation_config={
                    "temperature": 1,
                    "top_p": 0.95,
                    "max_output_tokens": 8192,
                    "response_mime_type": "text/plain",
                    },
                )
            self._print_debug("Fell back to non-cached model")

    def _check_context_relevance(self, question: str) -> bool:
        """Check if the current context can answer the new question."""
        # No context to answer the question
        if self.current_context == "":
            return False

        # Get conversation history from chat
        conversation_context = "\n".join(
            [f"Previous {msg.role}: {msg.parts[0].text}" for msg in self.chat.history[2:]]  # Skip system prompt
        ) if self.chat.history else ""

        prompt = f"""
        New question:
        "{question}"

        Previous conversation:
        {conversation_context}

        Currently loaded transcripts are from:
        {', '.join(self.current_dates)}

        Your task: Determine if we should use current transcripts or need new ones.
        BE CONSERVATIVE - prefer using current context unless absolutely necessary.

        Return USE_CURRENT_CONTEXT if ANY of these are true:
        - Question is a follow-up or clarification of previous discussion
        - Question uses words like "that", "this", "those", "they" referring to current context
        - Question asks for more details about topics/people already discussed
        - Question can be partially answered with current transcripts
        - Question is about interpretation or analysis of current content

        ONLY return NEED_NEW_CONTEXT if ALL of these are true:
        - Question explicitly asks about different dates/meetings
        - Question mentions specific people/topics definitely not in current transcripts
        - Question contains explicit search requests like "find all instances" or "search all transcripts"
        - Current context is completely irrelevant to the new question
        - You are 100% certain new transcripts are required

        Return ONLY one of these: USE_CURRENT_CONTEXT or NEED_NEW_CONTEXT
        """

        response = self.retrieval_model.generate_content(prompt)
        decision = response.text.strip()
        self._print_debug(self._format_usage_stats(response.usage_metadata))
        self._print_debug(f"Checking context relevance decision: {decision}")
        return decision == "USE_CURRENT_CONTEXT"

    #
    # FILTERING METHODS
    #

    def _select_relevant_transcripts(self, question: str) -> list[str]:
        """Select relevant transcripts based on question and available summaries."""

        prompt = f"""
        Given this question about the Ontario Legislature:
        "{question}"

        Available transcript dates range from {self.earliest_date} to {self.latest_date}.

        Using the transcript summaries provided in the context, select the most relevant dates that would help answer the question.
        CRITICAL: You MUST start with the MOST RECENT transcripts ({self.latest_date}) unless the question
        specifically asks about historical events or mentions older dates.

        Consider in order of priority:
        1. Most recent dates (starting from {self.latest_date})
        2. Explicit date references in the question
        3. Speakers mentioned
        4. Topics and their semantic similarities
        5. Bill discussions

        Return ONLY the dates, one per line, in the format YYYY-MM-DD.
        Limit your response to {self.MAX_TRANSCRIPT_CONTEXT} dates maximum.
        Start with the most recent relevant dates.
        """

        # get most recent relevant
        response = self.retrieval_model.generate_content(prompt)
        response_text = response.text
        relevant_dates = [
            date for date in response_text.split("\n")
            if date.strip() in self.available_dates
        ]
        # fallback: if none returned
        if not relevant_dates:
            self._print_debug("No specific dates found, using most recent transcripts")
            relevant_dates = self.available_dates[:self.MAX_TRANSCRIPT_CONTEXT]

        relevant_dates = sorted(relevant_dates, reverse=True)[:self.MAX_TRANSCRIPT_CONTEXT]

        self._print_debug(self._format_usage_stats(response.usage_metadata))
        self._print_debug(f"Selected dates: {relevant_dates}")
        return relevant_dates


    #
    # RESPONSE GENERATION METHODS
    #
    def _generate_response(self, question: str) -> Union[str, Generator]:
        """
        Generate response using selected transcripts.
        Returns a string if not streaming, otherwise a generator.
        """

        try:
            response = self.chat.send_message(question, stream=self.streaming)
            self._print_debug("Generated response")  # Add debug logging

            if not self.streaming:
                response_text = response.text
                return response_text

            return response

        except Exception as e:
            self._print_debug(f"Error generating response: {str(e)}")
            raise

    def chat_interface(self, question: str) -> Union[str, Generator]:
        """Main chat interface."""
        try:
            # Check if current context is relevant
            if self._check_context_relevance(question):
                self._print_debug("Using cached context")
                response = self._generate_response(question)
            else:
                self._print_debug("Fetching new context")
                relevant_dates = self._select_relevant_transcripts(question)
                self._update_current_context(relevant_dates)

                if not relevant_dates:
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

        except Exception as e:  # FIXME what kind of exception?
            return f"{Fore.RED}Sorry, I encountered an error: {e!s}{Style.RESET_ALL}"


def main():
    # Initialize bot
    bot = OLABot("../citizengemini/hansard.json", debug=True)
    bot.print_welcome()

    try:
        while True:
            question = input(f"{Fore.GREEN}Your question: {Style.RESET_ALL}")
            if question.lower() == "quit":
                break

            bot.print_question(question)
            response = bot.chat_interface(question)
            bot.print_response(response)
    except KeyboardInterrupt:
        bot._print_debug(f"Interrupted by user.")
    finally:
        try:
            bot.current_context_cache.delete()
            bot.transcript_summaries_cache.delete()
            bot._print_debug("Deleted all caches")
        except Exception as e:
            bot._print_debug(f"Error cleaning up caches: {e}")

        print(f"\n{Fore.CYAN}Goodbye! 👋{Style.RESET_ALL}\n")

if __name__ == "__main__":
    main()
