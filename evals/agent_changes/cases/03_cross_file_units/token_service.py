import time

from token_model import Token


def is_active(token: Token) -> bool:
    return token.expires_at_ms > time.time()
