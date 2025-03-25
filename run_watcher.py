import asyncio
from agents import drive_watcher

async def main():
    """Run the DriveWatcher as a standalone process."""
    try:
        print("\n=== Starting DriveWatcher Process ===")
        await drive_watcher.check_for_changes()
    except Exception as e:
        print(f"Error in DriveWatcher: {e}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())
