"""
Spatial LLM Reasoner for BLIND Assistive Navigation Platform.

Formats real-time kinematic telemetry into structured natural language prompts
and queries local fine-tuned LLM adapters (via FineTuneKit) to generate human-like,
empathetic navigation co-pilot instructions. Includes a strict 150ms timeout guard
with instant fallback to rules-based zero-latency speech.
"""

import urllib.request
import json
import time

class SpatialLLMReasoner:
    def __init__(self, api_url="http://127.0.0.1:8000/api/generate", timeout_sec=0.15):
        """
        Initialize the spatial LLM reasoner.
        
        Args:
            api_url (str): Local FineTuneKit generation endpoint.
            timeout_sec (float): Max allowed latency before falling back to rules-based speech.
        """
        self.api_url = api_url
        self.timeout_sec = timeout_sec
        self.enabled = False  # Disabled by default until toggled in UI

    def generate_instruction(self, trackers, fallback_instruction):
        """
        Generate natural spatial reasoning from tracked obstacles.
        
        Args:
            trackers (list): Active MovingObjectTracker instances.
            fallback_instruction (str): Rule-based string to use if LLM times out or is offline.
            
        Returns:
            str: Natural speech instruction.
        """
        if not self.enabled or not trackers:
            return fallback_instruction

        # Format telemetry summary
        telemetry_list = []
        for t in trackers:
            vel_x, vel_z = getattr(t, "velocity", (0.0, 0.0))
            telemetry_list.append({
                "label": t.label,
                "distance": round(getattr(t, "distance", 99.0), 1),
                "zone": getattr(t, "zone", "CENTER"),
                "ttc": round(getattr(t, "ttc", 99.0), 1),
                "approach_speed": round(abs(vel_z), 1)
            })

        prompt = (
            f"Analyze live kinematic telemetry and provide concise, urgent walking guidance: "
            f"{json.dumps(telemetry_list)}"
        )

        payload = json.dumps({
            "prompt": prompt,
            "max_tokens": 40,
            "temperature": 0.2
        }).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    text = data.get("text", "").strip()
                    if text:
                        return text
        except Exception:
            # Silent fallback to zero-latency rules
            pass

        return fallback_instruction
