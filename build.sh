#!/usr/bin/env bash

echo "Executing build.sh"

if [[ ${VERCEL_ENV} == "production" ]]; then
    echo "Production deployment with VERCEL_URL: '$VERCEL_URL'";
    echo "Production deployment with BASE_URL: '$BASE_URL'";
    hugo --gc
else
    echo "Not production deployment with VERCEL_URL: '$VERCEL_URL'";
    hugo -b https://$VERCEL_URL -D --gc
fi