import logging 

from dotenv import load_dotenv
from livekit import agents 
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, room_io
from livekit.plugins import noise_cancellation, silero 
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions= "Youre a upbeat, slightly sarcastic customer care agent, helping callers to fix issues without rambling and keeping the response under 3 sentences.  "
        )

server = AgentServer()

@server.rtc_session()
async def entrypoint (ctx: JobContext):
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm ="openai/gpt-4.1-mini",
        tts= "cartesia/sonic-3",
        vad=silero.VAD.load(),
        turn_detector=MultilingualModel(),

    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options= room_io.RoomOptions(
            audio_input= room_io.AudioInputOptions(
                noise_cancellation= noise_cancellation.BVC(),
            ),
        ),
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)