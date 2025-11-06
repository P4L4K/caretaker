from sqlalchemy.orm import Session
from ..tables.token_blocklist import TokenBlocklist

class TokenBlocklistRepo:
    @staticmethod
    def add_token_to_blocklist(db: Session, token: str):
        blocklisted_token = TokenBlocklist(token=token)
        db.add(blocklisted_token)
        db.commit()

    @staticmethod
    def is_token_blocklisted(db: Session, token: str) -> bool:
        return db.query(TokenBlocklist).filter(TokenBlocklist.token == token).first() is not None
