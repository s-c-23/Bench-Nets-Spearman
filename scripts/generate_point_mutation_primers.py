import sys
import argparse
from pathlib import Path

import csv2libs


def cli(argv):
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-s", "--sequence",
						help = "The input gene nucleotide sequence",
						type = str)
	parser.add_argument("-p", "--prediction",
						help = "Neurel net prediction file",
						type = Path)
	parser.add_argument("-m", "--point_mutation_file",
						help = "Point mutation file, format: (WTAA)(Position)(Mutation), each on a new line \
								or the same line if you want them to clustered on the same primer",
						type = Path)
	parser.add_argument("-o", "--output",
						help = "Output CSV name",
						type = Path,
						default = "primers.csv")
	parser.add_argument("-c", "--chain_id",
						help = "Chain label of interest for the protein",
						type = str,
						default = "A")
	parser.add_argument("-ftm", "--forward_tm",
						help = "Melting temperature for the forward primer annealing 3' region",
						type = float,
						default = 65.0)
	parser.add_argument("-rtm", "--reverse_tm",
						help = "Melting temperature for reverse primer annealing 3' region",
						type = float,
						default = 65.0)
	parser.add_argument("-ml", "--mutagenesis_length",
						help = "Length of the mutagenesis region in BPs",
						type = int,
						default = 20)
	parser.add_argument("-tmt", "--tm_tolerance",
						help = "how much leeway to give in differences between set Tm and calculated Tm",
						type = float,
						default = 0.5)
	parser.add_argument("-sc", "--na_eq",
						help = "Sodium-equivalent (von Ahsen et al 2001 Clin Chem) salt correction (Na, K, Tris, Mg, dNTPS)",
						type = float,
						default = 50)


	if len(argv) == 0:
		print(parser.print_help())
		return
	
	args = parser.parse_args(argv)
	
	csv2libs.generate_point_mutation_primers(args.sequence, args.prediction, args.point_mutation_file, args.output, args.chain_id, args.mutagenesis_length, [args.forward_tm, args.reverse_tm], args.tm_tolerance, args.na_eq)


if __name__ == "__main__":
	cli(sys.argv[1:])


