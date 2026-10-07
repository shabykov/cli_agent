import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Optional, Any

from anthropic import AsyncAnthropic
from mcp import Client, StdioServerParameters, types
from mcp.client import ClientRequestContext
from mcp.shared.dispatcher import ProgressFnT
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    TextContent,
    SamplingMessage,
)

anthropic_client: Optional[AsyncAnthropic] = None
model = "claude-sonnet-4-5"


async def chat(input_messages: list[SamplingMessage], max_tokens=4000):
    messages = []
    for msg in input_messages:
        if msg.role == "user" and msg.content.type == "text":
            content = (
                msg.content.text
                if hasattr(msg.content, "text")
                else str(msg.content)
            )
            messages.append({"role": "user", "content": content})
        elif msg.role == "assistant" and msg.content.type == "text":
            content = (
                msg.content.text
                if hasattr(msg.content, "text")
                else str(msg.content)
            )
            messages.append({"role": "assistant", "content": content})
    # created lazily: main.py loads .env only after importing this module
    global anthropic_client
    if anthropic_client is None:
        anthropic_client = AsyncAnthropic()

    response = await anthropic_client.messages.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    text = "".join([p.text for p in response.content if p.type == "text"])
    return text


async def sampling_callback(
        context: ClientRequestContext, params: CreateMessageRequestParams
):
    # Call Claude using the Anthropic SDK
    text = await chat(params.messages)

    return CreateMessageResult(
        role="assistant",
        model=model,
        content=TextContent(type="text", text=text),
    )


async def print_summarize_progress_callback(
        progress: float, total: float | None, message: str | None
):
    if total is not None:
        percentage = (progress / total) * 100
        print(f"Summarize progress: {progress}/{total} ({percentage:.1f}%) {message or ''}")
    else:
        print(f"Summarize progress: {progress} {message or ''}")


async def print_research_progress_callback(
        progress: float, total: float | None, message: str | None
):
    if total is not None:
        percentage = (progress / total) * 100
        print(f"Research progress: {progress}/{total} ({percentage:.1f}%) {message or ''}")
    else:
        print(f"Research progress: {progress} {message or ''}")


class MCPClient:
    def __init__(
            self,
            command: str,
            args: list[str],
            env: Optional[dict] = None,
    ):
        self._command = command
        self._args = args
        self._env = env
        self._client: Optional[Client] = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._progress_callbacks: dict[str, ProgressFnT] = {
            "summarize": print_summarize_progress_callback,
            "research": print_research_progress_callback,
        }

    async def connect(self):
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )
        # Client negotiates the protocol version and answers the server's
        # sampling requests (InputRequiredResult) via sampling_callback
        self._client = await self._exit_stack.enter_async_context(
            Client(server_params, sampling_callback=sampling_callback)
        )

    def session(self) -> Client:
        if self._client is None:
            raise ConnectionError(
                "Client session not initialized or cache not populated. Call connect_to_server first."
            )
        return self._client

    async def list_tools(self) -> list[types.Tool]:
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any] | None = None,
            *,
            progress_callback: ProgressFnT | None = None,
    ) -> types.CallToolResult | None:
        return await self.session().call_tool(
            name,
            arguments,
            progress_callback=progress_callback or self._progress_callbacks.get(name),
        )

    async def list_prompts(self) -> list[types.Prompt]:
        result = await self.session().list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name, args: dict[str, str]):
        result = await self.session().get_prompt(prompt_name, args)
        return result.messages

    async def read_resource(self, uri: str) -> Any:
        result = await self.session().read_resource(uri)
        resource = result.contents[0]

        if isinstance(resource, types.TextResourceContents):
            if resource.mime_type == "application/json":
                return json.loads(resource.text)

            return resource.text

    async def cleanup(self):
        await self._exit_stack.aclose()
        self._client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


# For testing
async def main():
    async with MCPClient(
            # If using Python without UV, update command to 'python' and remove "run" from args.
            command="uv",
            args=["run", "mcp_server.py"],
    ) as _client:
        pass


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
