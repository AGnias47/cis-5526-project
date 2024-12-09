from requests.adapters import HTTPAdapter, Retry
from requests_ratelimiter import LimiterSession


def limiting_retrying_session():
    session = LimiterSession(per_second=10)
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) "
            "Gecko/20100101 Firefox/115.0"
        }
    )
    return session
