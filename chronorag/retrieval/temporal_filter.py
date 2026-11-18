import logging
from datetime import datetime
from typing import Optional, List, Tuple
from sqlalchemy.orm import Query
from chronorag.database.models import Claim
from chronorag.database.bitemporal import is_claim_valid_at, was_claim_recorded_at, is_bitemporally_active

logger = logging.getLogger(__name__)


class DeterministicTemporalFilter:
    """
    Applies strict deterministic SQL & memory filters for bitemporal eligibility.
    Guarantees that no claim outside valid time or transaction time is admitted.
    """

    @classmethod
    def apply_sql_filter(
        cls,
        query: Query,
        as_of_valid_time: Optional[datetime] = None,
        as_of_transaction_time: Optional[datetime] = None,
    ) -> Query:
        """Applies SQL WHERE constraints based on bitemporal boundaries."""
        # 1. Valid Time Filter
        if as_of_valid_time is not None:
            # Claim must have started at or before as_of_valid_time, or start is null (unbounded past)
            # AND must NOT have valid_from_unknown = True
            query = query.filter(
                Claim.valid_from_unknown == False,
                (Claim.valid_from <= as_of_valid_time) | (Claim.valid_from == None),
                (Claim.valid_until > as_of_valid_time) | (Claim.valid_until == None),
            )

        # 2. Transaction Time Filter (What the system knew at as_of_transaction_time)
        if as_of_transaction_time is not None:
            query = query.filter(
                Claim.recorded_from <= as_of_transaction_time,
                (Claim.recorded_until > as_of_transaction_time) | (Claim.recorded_until == None),
            )

        return query

    @classmethod
    def filter_claim_list(
        cls,
        claims: List[Claim],
        as_of_valid_time: Optional[datetime] = None,
        as_of_transaction_time: Optional[datetime] = None,
    ) -> List[Claim]:
        """In-memory deterministic filter enforcing half-open bitemporal interval math."""
        eligible: List[Claim] = []
        for c in claims:
            # Check valid time
            if as_of_valid_time is not None:
                if not is_claim_valid_at(
                    c.valid_from,
                    c.valid_until,
                    as_of_valid_time,
                    c.valid_from_unknown,
                    c.valid_until_unknown,
                ):
                    continue

            # Check transaction time
            if as_of_transaction_time is not None:
                if not was_claim_recorded_at(
                    c.recorded_from,
                    c.recorded_until,
                    as_of_transaction_time,
                ):
                    continue

            eligible.append(c)
        return eligible
