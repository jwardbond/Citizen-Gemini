import inspect
import json
import os
import re
from pathlib import Path
from typing import Union, Generator
import dotenv
import google.generativeai as genai
from google.generativeai import caching
from colorama import Back, Fore, Style

dotenv.load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class OLABot:
    def __init__(self, hansard_path: Path, debug: bool = False, streaming: bool = True):
        """Initialize the Ontario Legislature Assistant Bot."""
        if not isinstance(hansard_path, Path):
            hansard_path = Path(hansard_path)

        # Load transcripts
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

        # Initialize Gemini main model
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            },
        )

        # Initialize Gemini retrieval model
        # with transcript summaries as cached context
        self.transcript_summaries_cache = caching.CachedContent.create(
            model = 'models/gemini-1.5-flash-8b',
            display_name = 'OLABot Transcript Summaries',
            contents = [self.transcript_condensed]
        )
        self.retrieval_model = genai.GenerativeModel.from_cached_content(
            cached_content = self.transcript_summaries_cache,
            generation_config={
                "temperature": 1,
                "max_output_tokens": 1024,
                "response_mime_type": "text/plain",
            }
        )

        # Initialize Gemini small model -- used for checking current context relevance
        self.small_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-8b",
            generation_config={
                "temperature": 1,
                "max_output_tokens": 1024,
                "response_mime_type": "text/plain",
            },
        )

        # Initialize conversation history
        self.conversation_history = []

        # Initialize current transcript context
        self.current_dates = []
        self.current_context = ""

        # some constants
        self.MAX_CONVERSATION_HISTORY = 5  # Max number of previous Q/A pairs
        self.MAX_TRANSCRIPT_CONTEXT = 5  # The number of transcript files

        # whether in debug mode
        self.debug = debug
        self.streaming = streaming

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
    # CONTEXT METHODS
    #
    def _update_current_context(self, dates: list[str]) -> None:
        """Update the current transcript context."""
        # TODO this should update the context in gemini
        # TODO warning if dates > max transcript context
        self.current_dates = dates
        self.current_context = "\n\n".join(
            [
                f"**Transcript from {date}**:\n{self.transcripts[date]}"
                for date in dates[: self.MAX_TRANSCRIPT_CONTEXT]
            ],
        )

    def _check_context_relevance(self, question: str) -> bool:
        """Check if the current context can answer the new question."""
        # No context to answer the question
        if self.current_context == "":
            return False

        # Get recent conversation context
        recent_exchanges = self.conversation_history[-self.MAX_CONVERSATION_HISTORY :]
        conversation_context = "\n".join(
            [
                f"Previous Q: {exchange['question']}\nPrevious A: {exchange['response']}"
                for exchange in recent_exchanges
            ],
        )

        # Create a condensed representation of available transcripts
        transcript_content = "\n\n".join(
            [
                f"**{date}**:\n{self.transcript_summaries[date]}"
                for date in self.current_dates
            ],
        )

        prompt = f"""
        Given this new question about the Ontario Legislature:
        "{question}"

        Recent conversation history:
        {conversation_context}

        Current transcript summaries:
        {transcript_content}

        Analyze if the current transcripts can answer this question or if we need new ones. Use the following guidelines:

        1. USE_CURRENT_CONTEXT if:
           - Question follows naturally from conversation history
           - Asks for more details about current topics/speakers
           - References "that", "this", "they" referring to current context
           - Current transcripts contain relevant dates/people/topics
           - Different aspect of same events/discussions

        2. NEED_NEW_CONTEXT if:
           - Mentions new person not in current context
           - Asks about specific person across all meetings
           - Completely new topic unrelated to current transcripts
           - Explicit requests like "search all transcripts" or "forget that"
           - Different time period or different speakers
           - Current transcripts unlikely to contain complete answer
           - Contains phrases like "tell me more about", "all available meetings", "search for"


        Return ONLY one of these: USE_CURRENT_CONTEXT or NEED_NEW_CONTEXT
        """

        response = self.small_model.generate_content(prompt)
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

        Select the most relevant transcript dates from these summaries that would help answer the question.
        IMPORTANT: Strongly prefer the most recent transcripts unless there is a compelling reason to use older ones.

        Consider in order of priority:
        1. Most recent dates that contain relevant information
        2. Explicit date references in the question
        3. Speakers mentioned
        4. Topics and their semantic similarities
        5. Bill discussions

        Only use older transcripts if they contain significantly more relevant information than recent ones.

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
        # Get recent conversation context
        recent_exchanges = self.conversation_history[
            -self.MAX_CONVERSATION_HISTORY :
        ]  # Last exchanges
        conversation_context = "\n".join(
            [
                f"Previous Q: {exchange['question']}\nPrevious A: {exchange['response']}"
                for exchange in recent_exchanges
            ],
        )

        # TODO ideally (for rubric points) this is cached. Unsure how to include this in the prompt in that case though
        transcript_content = "\n\n".join(
            [f"**{date}**:\n{self.transcripts[date]}" for date in self.current_dates],
        )

        prompt = f"""
        You are a helpful assistant making the Ontario Legislature more accessible to average citizens.
        You are given a question and a set of transcripts from the Ontario Legislature.
        Be specific and concise in your responses. Do not talk about irrelevant things in the transcripts.
        Make sure to use all the information available to you to answer the question.

        Current Question: "{question}"

        {f'''Recent conversation history:
        {conversation_context}''' if recent_exchanges else ''}

        Guidelines for your response:
        1. Be concise and clear - avoid political jargon
        2. If this is a follow-up question, reference relevant information from previous responses
        3. If quoting from transcripts, only use the most relevant quotes
        4. Structure your response in short paragraphs
        5. Include the date when mentioning specific discussions
        6. If something is unclear or missing from the transcripts, say so

        Remember: Your audience is the average citizen who wants to understand what happened in their legislature.
        If the audience asks for more detail, feel free to provide a more comprehensive answer.

        Transcripts:
        {transcript_content}
        """

        response = self.model.generate_content(prompt, stream=self.streaming)

        if not self.streaming:
            response_text = response.text
            self.conversation_history.append(
                {"question": question, "response": response_text},
            )
            return response_text

        return response

    def chat(self, question: str) -> Union[str, Generator]:
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

            self.conversation_history.append(
                {"question": question, "response": full_response},
            )
            self._print_debug(self._format_usage_stats(chunk.usage_metadata))

        except Exception as e:  # FIXME what kind of exception?
            return f"{Fore.RED}Sorry, I encountered an error: {e!s}{Style.RESET_ALL}"


def main():
    # Initialize bot
    bot = OLABot("hansard.json", debug=True)
    bot.print_welcome()

    while True:
        question = input(f"{Fore.GREEN}Your question: {Style.RESET_ALL}")
        if question.lower() == "quit":
            print(f"\n{Fore.CYAN}Goodbye! 👋{Style.RESET_ALL}\n")
            break

        bot.print_question(question)
        response = bot.chat(question)
        bot.print_response(response)

if __name__ == "__main__":
    main()
