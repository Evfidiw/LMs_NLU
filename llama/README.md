## Steps

1. Process the data to obtain ``label.json`` and ``text.json``. For example:

    ```
    python llama/dataset/trec6/data_process.py
    ```

2. Generate the output. 

    ```
    sh llama_zs_trec6.sh
    ```
    
3. Evaluate. 

    ```
    python llama/dataset/trec6/evaluate.py
    ```

    