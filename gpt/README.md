## Steps

1. Process the data to obtain ``label.json`` and ``text.json``. For example:

    ```
    python gpt/dataset/trec6/data_process.py
    ```

2. Generate the output.

    ```
    sh gpt_zs_trec6.sh
    sh gpt_zs_sst2.sh
    ```

3. Evaluate. 

    ```
    python gpt/dataset/trec6/evaluate.py
    ```

    