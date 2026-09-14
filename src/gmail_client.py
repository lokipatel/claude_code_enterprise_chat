"""Gmail API client: OAuth handling, message fetching, and payload parsing."""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from pydantic import BaseModel

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class EmailMessage(BaseModel):
    """Structured representation of a parsed Gmail message."""

    message_id: str
    thread_id: str
    subject: str
    sender: str
    recipient: str | None = None
    date: datetime
    body_text: str


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text fallback for messages with no text/plain part."""

    _SKIP_TAGS = {"script", "style"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data.strip())

    def get_text(self) -> str:
        return "\n".join(self._chunks)


def _html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


class GmailAuthError(RuntimeError):
    """Raised when OAuth authentication with Gmail fails."""


class GmailClient:
    """Thin wrapper around the Gmail API for reading email threads."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._service: Resource | None = None

    # -- Auth -----------------------------------------------------------

    def authenticate(self) -> Resource:
        """Authenticate via OAuth, caching/refreshing the token in token.json."""
        if self._service is not None:
            return self._service

        creds: Credentials | None = None
        token_path = self._settings.gmail_token_path
        creds_path = self._settings.gmail_credentials_path
        scopes = list(self._settings.gmail_scopes)

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except RefreshError as exc:
                    raise GmailAuthError(
                        f"Failed to refresh cached Gmail token: {exc}"
                    ) from exc
            else:
                if not creds_path.exists():
                    raise GmailAuthError(
                        f"Gmail OAuth credentials not found at {creds_path}. "
                        "Place the downloaded credentials.json there first."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), scopes)
                creds = flow.run_local_server(port=0)

            token_path.write_text(creds.to_json())

        self._service = build("gmail", "v1", credentials=creds)
        return self._service

    # -- Fetching ---------------------------------------------------------

    def fetch_latest_emails(
        self, max_results: int = 10, query: str = "label:INBOX"
    ) -> list[dict[str, Any]]:
        """Fetch raw Gmail message payloads matching `query`."""
        service = self.authenticate()

        list_response = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        message_refs = list_response.get("messages", [])

        raw_messages: list[dict[str, Any]] = []
        for ref in message_refs:
            raw = (
                service.users()
                .messages()
                .get(userId="me", id=ref["id"], format="full")
                .execute()
            )
            raw_messages.append(raw)

        return raw_messages

    # -- Parsing ------------------------------------------------------------

    @staticmethod
    def _decode_part_data(data: str) -> str:
        padded = data + "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")

    @classmethod
    def _extract_body(cls, payload: dict[str, Any]) -> str:
        """Walk the MIME tree, preferring text/plain and falling back to text/html."""
        plain_parts: list[str] = []
        html_parts: list[str] = []

        def walk(part: dict[str, Any]) -> None:
            mime_type = part.get("mimeType", "")
            body_data = part.get("body", {}).get("data")

            if body_data and mime_type == "text/plain":
                plain_parts.append(cls._decode_part_data(body_data))
            elif body_data and mime_type == "text/html":
                html_parts.append(cls._decode_part_data(body_data))

            for sub_part in part.get("parts", []) or []:
                walk(sub_part)

        walk(payload)

        if plain_parts:
            return "\n".join(plain_parts).strip()
        if html_parts:
            return _html_to_text("\n".join(html_parts)).strip()
        return ""

    @staticmethod
    def _get_header(headers: list[dict[str, str]], name: str) -> str | None:
        for header in headers:
            if header.get("name", "").lower() == name.lower():
                return header.get("value")
        return None

    @classmethod
    def parse_email_payload(cls, raw_message: dict[str, Any]) -> EmailMessage:
        """Convert a raw Gmail API message JSON into a structured EmailMessage."""
        payload = raw_message.get("payload", {})
        headers = payload.get("headers", [])

        subject = cls._get_header(headers, "Subject") or "(no subject)"
        sender = cls._get_header(headers, "From") or "(unknown sender)"
        recipient = cls._get_header(headers, "To")
        date_header = cls._get_header(headers, "Date")

        if date_header:
            date = parsedate_to_datetime(date_header)
        else:
            # Gmail's internalDate is epoch millis (UTC) and always present.
            date = datetime.fromtimestamp(int(raw_message["internalDate"]) / 1000)

        body_text = cls._extract_body(payload)

        return EmailMessage(
            message_id=raw_message["id"],
            thread_id=raw_message["threadId"],
            subject=subject,
            sender=sender,
            recipient=recipient,
            date=date,
            body_text=body_text,
        )
