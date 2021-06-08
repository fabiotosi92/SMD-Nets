for i in $(seq -f "%05g" 0 8); do
    zip_file=UnrealStereo4K_${i}.zip
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/${zip_file}
    unzip ${zip_file}
    rm ${zip_file}
done
