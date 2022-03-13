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
	parser.add_argument("-o", "--output",
						help = "Output CSV name",
						type = Path,
						default = "primers.csv")
	parser.add_argument("-c", "--chain_id",
						help = "Chain label of interest for the protein",
						type = str,
						default = "A")
	parser.add_argument("-l", "--library_size",
						help = "The size of the library",
						type = int,
						default = 10000)
	parser.add_argument("-htm", "--homology_tm",
						help = "Melting temperature for the forward primer homology region",
						type = float,
						default = 60.0)
	parser.add_argument("-atm", "--annealing_tm",
						help = "Melting temperature for forward primer annealing region",
						type = float,
						default = 62.0)
	parser.add_argument("-rtm", "--reverse_tm",
						help = "Melting temperature for reverse primer",
						type = float,
						default = 62.0)
	parser.add_argument("-ml", "--mutagenesis_length",
						help = "Length of the mutagenesis region in BPs",
						type = int,
						default = 39)
	parser.add_argument("-tmt", "--tm_tolerance",
						help = "how much leeway to give in differences between set Tm and calculated Tm",
						type = float,
						default = 1)
	parser.add_argument("-st", "--size_tolerance",
						help = "how much leeway to give in size differences (in terms of order of magnitude)",
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
	csv2libs.generate_ssm_primers(args.sequence, args.prediction, args.output, args.chain_id, args.library_size, args.size_tolerance, args.mutagenesis_length, [args.homology_tm, args.annealing_tm, args.reverse_tm], args.tm_tolerance, args.na_eq)


if __name__ == "__main__":
	cli(sys.argv[1:])


