import asyncio
from main import ShimmerStreamer

async def test_sniff():
    app = ShimmerStreamer('config/shimmer_config.json')
    app.initialize_components()
    # Ensure shimmer.shimmer_config has passive_connect true and manual_rfcomm true
    await app.shimmer_client.connect()
    result = await app.shimmer_client.passive_sniff(seconds=5, max_bytes=1024)
    print(result)

asyncio.run(test_sniff())