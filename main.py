"""
Kira - Dein persönlicher Sprachassistent
=========================================
So funktioniert Kira:
1. Du sprichst ins Mikrofon
2. Whisper wandelt deine Sprache in Text um
3. Claude (KI) denkt über deine Frage nach
4. ElevenLabs liest dir die Antwort vor

Starte mit: python main.py
"""

import os
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import anthropic
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# .env Datei laden - dort stehen deine geheimen API-Keys
# --------------------------------------------------------------------------
load_dotenv()


# ==========================================================================
# AUDIO-AUFNAHME - Nimmt deine Stimme über das Mikrofon auf
# ==========================================================================
class AudioRecorder:
    """Nimmt Audio vom Mikrofon auf und speichert es als WAV-Datei."""

    def __init__(self, sample_rate: int = 16000):
        # Sample-Rate = wie oft pro Sekunde der Ton gemessen wird
        # 16000 ist perfekt für Spracherkennung (Whisper mag das)
        self.sample_rate = sample_rate

    def record(self, duration: int = 5) -> str:
        """
        Nimmt Audio auf und gibt den Pfad zur WAV-Datei zurück.

        Args:
            duration: Wie viele Sekunden aufgenommen werden sollen.

        Returns:
            Pfad zur gespeicherten Audio-Datei.
        """
        print(f"\n🎤 Ich höre dir zu... ({duration} Sekunden)")

        # Audio aufnehmen - das sind einfach ganz viele Zahlen,
        # die den Ton beschreiben (wie eine Wellenform)
        audio_data = sd.rec(
            frames=int(duration * self.sample_rate),  # Anzahl der Messpunkte
            samplerate=self.sample_rate,
            channels=1,        # 1 = Mono (reicht für Sprache)
            dtype="float32",   # Zahlenformat für die Audio-Daten
        )

        # Warten bis die Aufnahme fertig ist
        sd.wait()

        # Lautstärke prüfen (hilft beim Debuggen)
        volume = np.abs(audio_data).mean()
        print(f"✅ Aufnahme fertig! (Lautstärke: {volume:.4f})")
        if volume < 0.001:
            print("⚠️ Sehr leise! Ist dein Mikrofon eingeschaltet?")

        # Audio als temporäre WAV-Datei speichern
        # (tempfile erstellt eine Datei, die sich später selbst löscht)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio_data, self.sample_rate)

        return temp_file.name


# ==========================================================================
# SPRACHE ZU TEXT - Whisper wandelt deine Sprache in geschriebenen Text um
# ==========================================================================
class SpeechToText:
    """Verwendet OpenAI Whisper um Sprache in Text umzuwandeln."""

    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: Größe des Whisper-Modells.
                - "tiny"  = schnell, aber ungenauer
                - "base"  = guter Kompromiss (empfohlen zum Start)
                - "small" = besser, braucht mehr RAM
                - "medium"= sehr gut, braucht viel RAM
                - "large" = am besten, braucht eine starke GPU
        """
        print(f"⏳ Lade Whisper-Modell '{model_size}'...")
        self.model = whisper.load_model(model_size)
        print("✅ Whisper ist bereit!")

    def transcribe(self, audio_path: str) -> str:
        """
        Wandelt eine Audio-Datei in Text um.

        Args:
            audio_path: Pfad zur WAV-Datei.

        Returns:
            Der erkannte Text als String.
        """
        print("🔄 Wandle Sprache in Text um...")

        # Audio mit soundfile laden (braucht kein ffmpeg!)
        audio_data, sample_rate = sf.read(audio_path)

        # In Float32 umwandeln (das Format das Whisper erwartet)
        audio_data = audio_data.astype(np.float32)

        # Whisper analysiert die Audio-Daten und erkennt den Text
        result = self.model.transcribe(audio_data, language="de")

        text = result["text"].strip()
        print(f"📝 Erkannt: \"{text}\"")

        return text


# ==========================================================================
# CLAUDE KI - Schickt deinen Text an Claude und bekommt eine Antwort
# ==========================================================================
class KiraAssistant:
    """Kommuniziert mit Claude (Anthropic API) für intelligente Antworten."""

    def __init__(self):
        # API-Key aus der .env Datei holen
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ ANTHROPIC_API_KEY fehlt! "
                "Kopiere .env.example zu .env und trage deinen Key ein."
            )

        # Anthropic-Client erstellen (das ist die Verbindung zur API)
        self.client = anthropic.Anthropic(api_key=api_key)

        # System-Prompt = die "Persönlichkeit" von Kira
        self.system_prompt = (
            "Du bist Kira, ein freundlicher und hilfreicher Sprachassistent. "
            "Du antwortest auf Deutsch, kurz und präzise. "
            "Halte deine Antworten unter 3 Sätzen, damit die Sprachausgabe "
            "nicht zu lange dauert. Sei freundlich und natürlich."
        )

        # Gesprächsverlauf speichern, damit Kira sich erinnern kann
        self.conversation_history: list[dict] = []

    def ask(self, user_text: str) -> str:
        """
        Schickt eine Nachricht an Claude und gibt die Antwort zurück.

        Args:
            user_text: Das was du gesagt hast (als Text).

        Returns:
            Claudes Antwort als Text.
        """
        print("🤔 Kira denkt nach...")

        # Deine Nachricht zum Verlauf hinzufügen
        self.conversation_history.append({
            "role": "user",
            "content": user_text,
        })

        # Nachricht an Claude schicken
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,  # Maximale Länge der Antwort
            system=self.system_prompt,
            messages=self.conversation_history,
        )

        # Antwort aus der Response rausholen
        assistant_text = response.content[0].text

        # Antwort auch zum Verlauf hinzufügen (damit Kira Kontext hat)
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_text,
        })

        print(f"💬 Kira: {assistant_text}")
        return assistant_text


# ==========================================================================
# SPRACHAUSGABE - ElevenLabs liest die Antwort laut vor
# ==========================================================================
class TextToSpeech:
    """Verwendet ElevenLabs um Text in natürliche Sprache umzuwandeln."""

    def __init__(self):
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ ELEVENLABS_API_KEY fehlt! "
                "Kopiere .env.example zu .env und trage deinen Key ein."
            )

        self.client = ElevenLabs(api_key=api_key)

        # Stimmen-ID aus .env laden
        # Wenn keine angegeben: "Alice" (gute deutsche Stimme)
        # Du findest Voice-IDs auf elevenlabs.io unter Voices → ... → Copy Voice ID
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "") or "Xb7hH8MSUJpSbSDYk0k2"
        print(f"🎙️ Voice-ID: {self.voice_id}")

    def speak(self, text: str) -> None:
        """
        Wandelt Text in Sprache um und spielt sie ab.

        Args:
            text: Der Text der vorgelesen werden soll.
        """
        print("🔊 Kira spricht...")

        # Text an ElevenLabs schicken und Audio zurückbekommen
        audio_generator = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id="eleven_multilingual_v2",  # Unterstützt Deutsch
            output_format="mp3_44100_128",
        )

        # Audio-Daten sammeln (kommt als Generator in kleinen Stücken)
        audio_bytes = b"".join(audio_generator)

        # Als temporäre Datei speichern und mit soundfile abspielen
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_file.write(audio_bytes)
        temp_file.close()

        # MP3 laden und abspielen
        audio_data, sample_rate = sf.read(temp_file.name)
        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()  # Warten bis fertig abgespielt

        # Temp-Datei löschen
        os.unlink(temp_file.name)


# ==========================================================================
# HAUPTPROGRAMM - Hier läuft alles zusammen
# ==========================================================================
def main():
    """Startet Kira - die Hauptschleife des Sprachassistenten."""

    print("=" * 50)
    print("  🌟 Willkommen bei Kira!")
    print("  Dein persönlicher Sprachassistent")
    print("=" * 50)

    # Alle Komponenten erstellen
    recorder = AudioRecorder()
    stt = SpeechToText(model_size="base")
    assistant = KiraAssistant()
    tts = TextToSpeech()

    print("\n✅ Kira ist bereit! Drücke Enter um zu sprechen.")
    print("   Tippe 'exit' um zu beenden.\n")

    # Endlosschleife - Kira hört immer wieder zu
    while True:
        # Warten bis der Nutzer Enter drückt
        user_input = input("▶ Enter drücken zum Sprechen (oder 'exit'): ")

        if user_input.lower() == "exit":
            print("\n👋 Tschüss! Bis zum nächsten Mal!")
            break

        try:
            # Schritt 1: Audio aufnehmen
            audio_path = recorder.record(duration=5)

            # Schritt 2: Sprache zu Text
            text = stt.transcribe(audio_path)

            # Leere Aufnahme überspringen
            if not text:
                print("⚠️ Ich konnte nichts verstehen. Versuch es nochmal!")
                continue

            # Schritt 3: Text an Claude schicken
            answer = assistant.ask(text)

            # Schritt 4: Antwort vorlesen
            tts.speak(answer)

            # Temporäre Audio-Datei aufräumen
            os.unlink(audio_path)

        except KeyboardInterrupt:
            print("\n\n👋 Tschüss!")
            break
        except Exception as e:
            print(f"\n❌ Fehler: {e}")
            print("   Versuch es nochmal!\n")


# Das hier bedeutet: Führe main() aus, wenn die Datei direkt gestartet wird
if __name__ == "__main__":
    main()
