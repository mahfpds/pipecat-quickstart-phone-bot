# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Twilio Phone Example.

This runs a simple voice AI bot you can connect to via Twilio.

Required services (env-configurable):
- STT: Deepgram (DEEPGRAM_API_KEY)
- LLM: OpenAI (OPENAI_API_KEY, OPENAI_MODEL) or local Ollama (OLLAMA=1, OLLAMA_MODEL, OLLAMA_BASE_URL)
- TTS: ElevenLabs (ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID)

Transport:
- Twilio websocket connection (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN optional but recommended)

Run:
    python bot.py -t twilio -x your-public-hostname.example.com
"""

import os
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

# LLM providers
from pipecat.services.openai.llm import OpenAILLMService
try:
    # Available if you installed: pip install "pipecat-ai[ollama]"
    from pipecat.services.ollama.llm import OLLamaLLMService  # noqa: F401
    _OLLAMA_AVAILABLE = True
except Exception:
    _OLLAMA_AVAILABLE = False

# ElevenLabs TTS (WebSocket)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

load_dotenv(override=True)


def build_llm():
    """Create an LLM service based on env vars.

    Defaults to OpenAI (gpt-4o-mini). Set OLLAMA=1 to use local Ollama.
    """
    use_ollama = os.getenv("OLLAMA", "").strip() in ("1", "true", "True", "yes", "on")

    if use_ollama:
        if not _OLLAMA_AVAILABLE:
            raise RuntimeError(
                "OLLamaLLMService not available. Install extras: pip install 'pipecat-ai[ollama]'"
            )
        model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        logger.info(f"Using Ollama LLM: model={model}, base_url={base_url}")
        return OLLamaLLMService(model=model, base_url=base_url)

    # OpenAI default
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        logger.warning("OPENAI_API_KEY not set; if you intended to use Ollama, set OLLAMA=1.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    logger.info(f"Using OpenAI LLM: model={model}")
    return OpenAILLMService(api_key=openai_key, model=model)


def build_tts():
    """Create ElevenLabs TTS (WebSocket)."""
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabs TTS.")

    # Known-good default voice to avoid 1008 policy violation errors
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # "Rachel"

    # 8 kHz matches PSTN/Twilio bandwidth nicely.
    tts = ElevenLabsTTSService(
        api_key=api_key,
        voice_id=voice_id,
        model=os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5"),
        sample_rate=int(os.getenv("AUDIO_OUT_SAMPLE_RATE", "8000")),
    )

    logger.info(f"Using ElevenLabs TTS: voice_id={voice_id}")
    return tts


async def run_bot(transport: BaseTransport):
    logger.info("Starting bot")

    # STT: Deepgram
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # TTS: ElevenLabs
    tts = build_tts()

    # LLM: OpenAI or Ollama
    llm = build_llm()

    # ---- Prompt / Persona ----
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly AI assistant. Respond naturally and keep your answers conversational."
            ),
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),                 # Transport user input
            rtvi,                              # RTVI processor
            stt,                               # STT
            context_aggregator.user(),         # User responses
            llm,                               # LLM
            tts,                               # TTS (ElevenLabs)
            transport.output(),                # Transport bot output
            context_aggregator.assistant(),    # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=int(os.getenv("AUDIO_IN_SAMPLE_RATE", "8000")),
            audio_out_sample_rate=int(os.getenv("AUDIO_OUT_SAMPLE_RATE", "8000")),
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        messages.append(
            {"role": "system", "content": "Say hello and briefly introduce yourself."}
        )
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Auto-detected transport: {transport_type}")

    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
