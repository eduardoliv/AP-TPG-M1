def process_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()  # Remove espaços em branco extras
            if len(line) > 2:
                outfile.write(line[1:-1] + '\n')
            else:
                outfile.write('\n')  # Se a linha tiver menos de 2 caracteres, escreve uma linha vazia

# Exemplo de uso
input_file = 'clean_output_datasets/dataset2_stor_outputs.csv'  # Substitua pelo nome do seu arquivo de entrada
output_file = 'dataset2_stor_outputs.csv'   # Nome do arquivo de saída
process_lines(input_file, output_file)
