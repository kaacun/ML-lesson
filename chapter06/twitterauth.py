# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import sys

CONSUMER_KEY = "O4Tf7UWCI2MxNRLJDsU2LnxkV"
CONSUMER_SECRET = "llgGJ0mOD564PFM56sxDbMJkGMyPDDoEssyqxzGdQEtDUQ9WGK"

ACCESS_TOKEN_KEY = "140310678-9kbPy1ByxUuhRrWowBS2AWerf2HNr8zz8RcSkv6f"
ACCESS_TOKEN_SECRET = "Lw5BKzGrxjXdswoqk6uGy2AGTkMBZbjSIwgvy0ZH53OTe"

if CONSUMER_KEY is None or CONSUMER_SECRET is None or ACCESS_TOKEN_KEY is None or ACCESS_TOKEN_SECRET is None:
    print("""\
When doing last code sanity checks for the book, Twitter
was using the API 1.0, which did not require authentication.
With its switch to version 1.1, this has now changed.

It seems that you don't have already created your personal Twitter
access keys and tokens. Please do so at
https://dev.twitter.com/docs/auth/tokens-devtwittercom
and paste the keys/secrets into twitterauth.py

Sorry for the inconvenience,
The authors.""")

    sys.exit(1)
