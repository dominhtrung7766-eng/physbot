import asyncio
import edge_tts

async def test():
    comm = edge_tts.Communicate("Xin chào, tôi là PhysBot.", voice="vi-VN-HoaiMyNeural")
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            print(f"OK - nhận được {len(chunk['data'])} bytes audio")
            return
    print("FAIL - không có audio")

asyncio.run(test())