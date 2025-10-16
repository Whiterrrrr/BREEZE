#!/bin/bash

S3_URL=https://dl.fbaipublicfiles.com/exorl

DOMAINS=("walker" "jaco" "quadruped")
ALGOS=("rnd" "aps" "proto" "diayn")

for DOMAIN in "${DOMAINS[@]}"; do
    for ALGO in "${ALGOS[@]}"; do
        DIR=./datasets/${DOMAIN}
        mkdir -p ${DIR}/${ALGO}

        URL=${S3_URL}/${DOMAIN}/${ALGO}.zip

        echo "=============================================================================="

        if [ -d "${DIR}/${ALGO}" ] && [ "$(ls -A ${DIR}/${ALGO})" ]; then
            echo "Dataset ${DOMAIN}/${ALGO} already exists, skipping..."
            continue
        fi

        if wget ${URL} -P ${DIR}; then
            echo "Successfully downloaded ${ALGO}.zip for ${DOMAIN}"
            echo "Unzipping ${ALGO}.zip into ${DIR}/${ALGO}..."
            
            unzip -q ${DIR}/${ALGO}.zip -d ${DIR}/${ALGO}
            
            rm ${DIR}/${ALGO}.zip

            echo "Executing reformatting for ${DOMAIN}/${ALGO}..."
            python exorl_reformatter.py "${DOMAIN}_${ALGO}"
            
            echo "Completed processing ${DOMAIN}/${ALGO}"
        else
            echo "Failed to download ${ALGO}.zip for ${DOMAIN}"
            rm -f ${DIR}/${ALGO}.zip
        fi
        
        echo "=============================================================================="
        echo
    done
done

echo "All downloads completed!"