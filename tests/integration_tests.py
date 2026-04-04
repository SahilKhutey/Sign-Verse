import time
import torch
import numpy as np
import os
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.translation_engine import SignVerseTranslationEngine

class TestSignVerseEngine:
    """Integration and performance tests for the complete engine"""
    
    def __init__(self):
        self.engine = SignVerseTranslationEngine()
    
    async def test_sign_to_speech_pipeline(self):
        """Test complete sign → text → speech pipeline"""
        print("Testing Sign -> Speech Pipeline...")
        # Mock video (10 frames of 150-dim features)
        video = torch.randn(10, 150)
        
        # Run pipeline
        result = await self.engine.process_frame({"start_time": time.time(), "frames_processed": 0}, b"")
        
        # Verify text
        print(f"  Result Text: {result['text']}")
        assert result["text"] is not None
        
        # Verify speech is generated
        assert result["speech"] is not None
        assert len(result["speech"]) > 0
        print("  Sign -> Speech OK.")
    
    async def test_real_time_latency(self):
        """Test real-time performance"""
        print("Testing Real-time Latency...")
        latency_measurements = []
        test_frame = b"\x00" * 1024
        session = {"start_time": time.time(), "frames_processed": 0}
        
        for i in range(50):
            start = time.perf_counter()
            await self.engine.process_frame(session, test_frame)
            end = time.perf_counter()
            
            latency = (end - start) * 1000  # Convert to ms
            latency_measurements.append(latency)
        
        avg_latency = np.mean(latency_measurements)
        print(f"  Avg Latency: {avg_latency:.2f}ms")
        assert avg_latency < 200  # Should be under 200ms
        
        p95_latency = np.percentile(latency_measurements, 95)
        print(f"  P95 Latency: {p95_latency:.2f}ms")
        assert p95_latency < 300 
        print("  Latency OK.")

async def run_integration_tests():
    tester = TestSignVerseEngine()
    await tester.test_sign_to_speech_pipeline()
    await tester.test_real_time_latency()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_integration_tests())
