from rich.console import Console


class Logger:
    """Centralized Rich console logger with error, warning, and info levels."""

    def __init__(self) -> None:
        self._console = Console()

    @property
    def console(self) -> Console:
        return self._console

    def error(self, msg: str) -> None:
        self._console.print(f"[red]Error:[/red] {msg}")

    def warning(self, msg: str) -> None:
        self._console.print(f"[yellow]Warning:[/yellow] {msg}")

    def info(self, msg: str) -> None:
        self._console.print(f"[dim]{msg}[/dim]")

    def status(self, msg: str):
        return self._console.status(msg)


logger = Logger()
