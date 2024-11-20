import json
from datetime import datetime
import google.generativeai as genai
from colorama import init, Fore, Style, Back
import re
import os
import dotenv

dotenv.load_dotenv()
init()  # Initialize colorama

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

class OLABot:
    def __init__(self, hansard_path, debug=False):
        """Initialize the Ontario Legislature Assistant Bot"""
        # Load transcripts
        with open(hansard_path, 'r') as f:
            self.transcripts = json.load(f)
        # Get the date range of available transcripts
        self.available_dates = sorted(list(self.transcripts.keys()), reverse=True)
        self.earliest_date = self.available_dates[-1]
        self.latest_date = self.available_dates[0]

        self.transcript_speakers, self.transcript_topics, self.transcript_bills, self.transcript_summaries =\
            self.generate_transcript_summaries()

        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)

        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain"
            }
        )
        self.small_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-8b",
            generation_config={
                "temperature": 1,
                "max_output_tokens": 1024,
                "response_mime_type": "text/plain"
            }
        )

        # Initialize conversation history
        self.conversation_history = []

        # Initialize current transcript context
        self.current_dates = []
        self.current_content = ''

        # some constants
        self.MAX_CONVERSATION_HISTORY = 5
        self.MAX_TRANSCRIPT_CONTEXT = 5

        # whether in debug mode
        self.debug = debug

    # TRANSCRIPT SUMMARIES
    def generate_transcript_summaries(self):
        """Process and store topics and speakers for each transcript"""
        transcript_topics = {}
        transcript_speakers = {}
        transcript_bills = {}
        for date, transcript in self.transcripts.items():
            toc_lines = transcript.split('\n\n')[0:50]
            # Skip header lines and empty lines
            topics = [
                line.strip() for line in toc_lines
                if line.strip()
                and 'LEGISLATIVE ASSEMBLY' not in line
                and 'ASSEMBLÉE LÉGISLATIVE' not in line
                and not line.startswith('Thursday')
                and not line.startswith('Monday')
                and not line.startswith('Tuesday')
                and not line.startswith('Wednesday')
                and not line.startswith('Friday')
                and not line.startswith('Jeudi')
                and not line.startswith('Lundi')
                and not line.startswith('Mardi')
                and not line.startswith('Mercredi')
                and not line.startswith('Vendredi')
            ]
            transcript_topics[date] = topics

            # Find speakers in Question Period and other sections
            # Look for patterns like:
            # "Mr. Smith:", "Ms. Jones:", "Hon. Doug Ford:", "The Speaker:"
            # This check is imperfect, but it's a good enough heuristic
            # And we will toss it into LLM anyways
            '''
            Example of failure mode (has colon and starts with The):
            The 911 model of care that we referenced at the Association of Municipalities of Ontario conference earlier this week has been embraced: community paramedicine that allows community paramedics to go into those homes, for individuals who are able, in most cases with very little support, to stay safely in their home. The municipalities that have embraced that 911 model of care have loved it. In fact, our satisfaction rate, I believe, is in the 97th percentile.
            '''
            speakers = set()
            lines = transcript.split('\n')
            for line in lines:
                # Process lines that start with a title prefix
                if any(line.strip().startswith(prefix) for prefix in ['Mr.', 'Ms.', 'Mrs.', 'Hon.', "The"]):
                    if ':' in line:
                        speaker = line.split(':')[0].strip()
                        if ('(' in speaker and ')' in speaker) or any(title in speaker for title in ['Mr.', 'Ms.', 'Mrs.', 'Hon.']):
                            speakers.add(speaker)

            transcript_speakers[date] = list(speakers)

            # Find bills
            # Bills have a simple pattern: "Bill 123A"
            bills = set()
            bill_pattern = re.compile(r'Bill\s+\d+[A-Za-z]*')

            for line in lines:
                bill_matches = bill_pattern.findall(line)
                bills.update(bill_matches)

            transcript_bills[date] = list(bills)

        transcript_summaries = {
            date: f"{date} -- Speakers: {', '.join(transcript_speakers[date])} | Topics: {', '.join(transcript_topics[date])} | Bills: {', '.join(transcript_bills[date]) if date in transcript_bills else 'None'}"
            for date in self.available_dates
        }

        return transcript_topics, transcript_speakers, transcript_bills, transcript_summaries

    # PRINTING METHODS
    def print_debug(self, message):
        """Print debug messages in dim yellow"""
        if self.debug:
            print(f"{Fore.YELLOW}{Style.DIM}Debug - {message}{Style.RESET_ALL}")

    def print_usage_stats(self, usage_stats, operation_name=""):
        """
        Prints formatted usage statistics for a Gemini API call using print_debug.

        Args:
            usage_stats: The usage_metadata from a Gemini response
            operation_name: Optional name of the operation (e.g., "Routing", "Context Selection")
        """
        stats_message = (f"{operation_name} usage stats:\n"
                        f"\tprompt_token_count: {usage_stats.prompt_token_count}\n"
                        f"\tcandidates_token_count: {usage_stats.candidates_token_count}\n"
                        f"\ttotal_token_count: {usage_stats.total_token_count}")
        self.print_debug(stats_message)

    def print_welcome(self):
        """Print welcome message"""
        print(f"\n{Back.BLUE}{Fore.WHITE} CITIZEN GEMINI {Style.RESET_ALL}")
        print(f"{Fore.CYAN}Your AI Guide to the Ontario Legislature{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Type 'quit' to exit{Style.RESET_ALL}\n")

    def print_question(self, question):
        """Print user question with styling"""
        print(f"\n{Fore.GREEN}❓ You: {Style.BRIGHT}{question}{Style.RESET_ALL}")

    def print_response(self, response):
        """Print bot response with styling"""
        print(f"\n{Fore.BLUE}🤖 Assistant: {Style.NORMAL}{response}{Style.RESET_ALL}\n")
        print(f"{Style.DIM}---{Style.RESET_ALL}")  # Separator line


    # CONTEXT METHODS

    def update_current_context(self, dates):
        """Update the current transcript context"""
        self.current_dates = dates
        self.current_content = "\n\n".join([
            f"**Transcript from {date}**:\n{self.transcripts[date]}"
            for date in dates[:self.MAX_TRANSCRIPT_CONTEXT]
        ])

    def check_context_relevance(self, question):
        """
        Check if the current transcript context can answer the new question,
        taking into account the recent conversation history.
        """
        # No context to answer the question
        if self.current_content == '':
            return False

        # Get recent conversation context
        recent_exchanges = self.conversation_history[-self.MAX_CONVERSATION_HISTORY:]
        conversation_context = "\n".join([
        f"Previous Q: {exchange['question']}\nPrevious A: {exchange['response']}"
            for exchange in recent_exchanges
        ])

        # Create a condensed representation of available transcripts
        transcript_content = "\n\n".join([
            f"**{date}**:\n{self.transcript_summaries[date]}"
            for date in self.current_dates
        ])

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
        self.print_usage_stats(response.usage_metadata, "Checking context relevance")
        decision = response.text.strip()
        self.print_debug(f"Checking context relevance decision: {decision}")
        return decision == 'USE_CURRENT_CONTEXT'

    # ROUTING METHODS

    def route_question(self, question):
        """Determine what type of question and which transcripts to load"""
        prompt = f"""
        Analyze this question about the Ontario Legislature:
        "{question}"

        Return in JSON format:
        {{
            "type": "TOPIC_SEARCH or PERSON_STATEMENT or BILL_DISCUSSION",
            "topics": ["main topic", "related topics"],
            "time_period": {{
                "start": "YYYY-MM-DD if explicitly mentioned, otherwise null",
                "end": "YYYY-MM-DD if explicitly mentioned, otherwise null"
            }},
            "people": ["only names specifically mentioned in the question"],
            "bill_number": "only if a specific bill (like 'Bill 123') is mentioned, otherwise null"
        }}

        Important rules:
        - Only include dates if they are explicitly mentioned in the question
        - Only include people who are specifically named in the question
        - Only include bill numbers that are explicitly mentioned (format: 'Bill ###')
        - Do not infer or make up any information not directly stated in the question
        - Leave fields as null if the information isn't explicitly provided

        Do not include any markdown formatting or backticks in your response.
        """

        response = self.small_model.generate_content(prompt)
        self.print_usage_stats(response.usage_metadata, "Routing")
        response_text = response.text
        try:
            # Clean the response text
            response_text = response_text.strip()
            # Remove markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '', 1)
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            routing = json.loads(response_text)
            routing['topics'] = routing.get('topics', [])
            routing['people'] = routing.get('people', [])
            routing['bill_number'] = routing.get('bill_number', [])

            self.print_debug(f"Final routing: {routing}")  # Debug print
            return routing

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {response_text}")
            return {
                "type": "TOPIC_SEARCH",
                "topics": [question],
                "time_period": {
                    "start": self.earliest_date,
                    "end": self.latest_date
                },
                "people": [],
                "bill_number": None
            }

    # FILTERING METHODS

    def select_relevant_transcripts(self, routing_analysis):
        """Select relevant transcripts based on question and available transcripts"""
        dates = self.available_dates

        # Apply date filter if specified
        # Add date validation function
        def is_valid_date_format(date_str):
            return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date_str))

        # Filter transcripts by date if valid date format
        if routing_analysis.get('time_period'):
            start_date = routing_analysis['time_period'].get('start')
            end_date = routing_analysis['time_period'].get('end')

            if start_date and is_valid_date_format(start_date):
                dates = [d for d in dates if d >= start_date]
            if end_date and is_valid_date_format(end_date):
                dates = [d for d in dates if d <= end_date]

        # Create a condensed representation of available transcripts
        transcript_content = "\n\n".join([
            f"**{date}**:\n{self.transcript_summaries[date]}"
            for date in dates
        ])

        prompt = f"""
        Question type: {routing_analysis['type']}
        Topics mentioned: {', '.join(routing_analysis.get('topics', []))}
        People mentioned: {', '.join(routing_analysis.get('people', []))}
        Bill number: {routing_analysis.get('bill_number')}

        From these transcript summaries, return all relevant dates that would be useful to answer the question.
        Consider:
        1. Speakers mentioned in the question
        2. Topics and their semantic similarities
        3. Bill discussions if specified
        4. Most recent dates if everything else is equal

        Return ONLY the dates, one per line, in the format YYYY-MM-DD.

        Available transcripts:
        {transcript_content}
        """

        response = self.small_model.generate_content(prompt)
        self.print_usage_stats(response.usage_metadata, "Selecting relevant dates")
        response_text = response.text
        relevant_dates = [date for date in response_text.split('\n') if date in dates][:self.MAX_TRANSCRIPT_CONTEXT]

        self.print_debug(f"Selected dates: {relevant_dates}")
        return relevant_dates

    # RESPONSE GENERATION METHODS

    def generate_response(self, question):
        """Generate response using selected transcripts"""
        # Get recent conversation context
        recent_exchanges = self.conversation_history[-self.MAX_CONVERSATION_HISTORY:]  # Last exchanges
        conversation_context = "\n".join([
            f"Previous Q: {exchange['question']}\nPrevious A: {exchange['response']}"
            for exchange in recent_exchanges
        ])

        transcript_content = "\n\n".join([
            f"**{date}**:\n{self.transcripts[date]}"
            for date in self.current_dates
        ])

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

        response = self.model.generate_content(prompt)
        self.print_usage_stats(response.usage_metadata, "Generating response")
        response_text = response.text

        # Store this exchange in history
        self.conversation_history.append({
            'question': question,
            'response': response_text
        })

        return response_text

    def chat(self, question):
        """Main chat interface"""
        try:
            # Check if current context is relevant
            if self.check_context_relevance(question):
                self.print_debug("Using cached context")
                return self.generate_response(question)
            else:
                self.print_debug("Fetching new context")
                routing = self.route_question(question)
                relevant_dates = self.select_relevant_transcripts(routing)
                self.update_current_context(relevant_dates)

            if not relevant_dates:
                return "I couldn't find any relevant discussions in the available transcripts."

            return self.generate_response(question)

        except Exception as e:
            return f"{Fore.RED}Sorry, I encountered an error: {str(e)}{Style.RESET_ALL}"


def main():
    # Initialize bot
    bot = OLABot('hansard.json', debug=True)
    bot.print_welcome()

    while True:
        question = input(f"{Fore.GREEN}Your question: {Style.RESET_ALL}")
        if question.lower() == 'quit':
            print(f"\n{Fore.CYAN}Goodbye! 👋{Style.RESET_ALL}\n")
            break

        bot.print_question(question)
        response = bot.chat(question)
        bot.print_response(response)

if __name__ == "__main__":
    main()