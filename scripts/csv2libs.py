from pathlib import Path
import sys
import argparse
from collections import namedtuple, defaultdict
from typing import List, Tuple
import math
from abc import ABC, abstractmethod

import gemmi
import pandas as pd

from Bio.Data.CodonTable import standard_dna_table
from Bio.PDB.Polypeptide import three_to_one
from Bio.Seq import Seq
from Bio.SeqUtils.MeltingTemp import Tm_NN, salt_correction
from Bio import pairwise2


FT = standard_dna_table.forward_table
FT["TAA"] = "-"
FT["TAG"] = "-"
FT["TGA"] = "-"
BT = defaultdict(list)

bases = {
	"A":["A"],
	"G":["G"],
	"C":["C"],
	"T":["T"],
	"R":["A", "G"],
	"Y":["C", "T"],
	"M":["A", "C"],
	"K":["G", "T"],
	"S":["G", "C"],
	"W":["A", "T"],
	"H":["A", "C", "T"],
	"B":["G", "C", "T"],
	"V":["A", "C", "G"],
	"D":["A", "G", "T"],
	"N":["A", "C", "G", "T"]
}

AAs = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

for key, value in FT.items():
	BT[value].append(key)

def aas_from_codon(codon):
	possible_phenotypes = []
	for a in bases[codon[0]]:
		for b in bases[codon[1]]:
			for c in bases[codon[2]]:
				new_codon = a+b+c
				possible_phenotypes.append(FT[new_codon])

	return sorted(list(set(possible_phenotypes)))

def codons_from_codon(codon: str) -> str:
	possible_codons = []
	for a in bases[codon[0]]:
		for b in bases[codon[1]]:
			for c in bases[codon[2]]:
				new_codon = a+b+c
				possible_codons.append(new_codon)

	return possible_codons

def reverse_compliment(sequence: str):
	sequence = sequence.upper()
	pair = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
	rev_c = [pair[nt] for nt in sequence]
	rev_c.reverse()
	return ''.join(rev_c)


class NNOutput():
	"""
	A container class for the neural net output CSVs, converts them into a Pandas Dataframe for easy manipulation

	"""
	def __init__(self, csv_name: str):
		"""
		Initializes a NNOutput from a given path to CSV
		"""
		with open(csv_name, "rb") as f:
			self._df = pd.read_csv(f, index_col = 0)
		self.row = 0
		self.name = csv_name

	def __iter__(self):
		return self

	def __next__(self):
		if "pos" in self._df.columns:
			pos_column = "pos"
		else:
			pos_column = "position"

		if self.row == len(self._df) - 1:
			self.row = 0
			raise StopIteration
		else:
			row = self._df.iloc[self.row]

			pred = Prediction(row["chain_id"], row[pos_column], three_to_one(row["wtAA"]), sorted([(three_to_one(AA), row[f"pr{AA}"]) for AA in AAs], key = lambda x: x[1], reverse = True), float(row["avg_log_ratio"]))
			self.row += 1
			return pred

	def get_prediction(self, chain_id: str, position: int):
		if "pos" in self._df.columns:
			pos_column = "pos"
		else:
			pos_column = "position"

		row = self._df[(self._df["chain_id"] == chain_id) &  (self._df[pos_column] == position)].squeeze()
		assert not row.empty
		pred = Prediction(row["chain_id"], row[pos_column], three_to_one(row["wtAA"]), sorted([(three_to_one(AA), row[f"pr{AA}"]) for AA in AAs], key = lambda x: x[1], reverse = True), float(row["avg_log_ratio"]))
		return pred

	def determine_mutable_residues(self, chosen_chain_id: str, misprediction_cutoff: float, cum_prob_cutoff: float):
		mutable_residues = []
		for idx, pred in enumerate(self):
			if pred.chain_id != chosen_chain_id:
				continue	

			if pred.wtAA_prediction < misprediction_cutoff:
				aa_predictions = pred.extract_residues_within_cutoff(cum_prob_cutoff)
				aa_predictions.append(pred.wtAA)
				pred.determine_best_codon(aa_predictions)
				mutable_residues.append(pred)
		return mutable_residues

class Prediction():
	"""
	A container class for a Prediction

	"""
	def __init__(self, chain_id: str, position: int, wtAA: str, predictions: Tuple[str, float], score: float):
		self.chain_id = chain_id
		self.position = position
		self.wtAA = wtAA
		self.predictions = predictions
		self.wtAA_prediction = [prediction[1] for prediction in predictions if prediction[0] == wtAA][0]
		self.score = score
		self.best_codon = None
		self.size = None
		self.efficiency	 = None
		self.point_mutation = self._get_point_mutation(0)

	def __str__(self) -> str:
		return f"{self.chain_id}{self.position}"

	def __lt__(self, other):
		return self.position < other.position

	def __eq__(self, other):
		if not self.best_codon:
			return self.position == other.position 
		else:
			return (self.position == other.position) and (self.best_codon == other.best_codon) 


	def extract_residues_within_cutoff(self, cum_prob_cutoff: float):
		workable_predictions = self.predictions.copy()
		rep_residues = []
		cum_prob = 0

		while cum_prob < cum_prob_cutoff and len(workable_predictions) > 0:
			residue, probability = workable_predictions.pop(0)
			rep_residues.append(residue)
			cum_prob += probability

		return rep_residues


	def _solve_codon_scheme(self, codon_predictions):
		all_solutions = []
		self._scs_helper("", 1, codon_predictions, all_solutions)
		return all_solutions

	def _scs_helper(self, codon, site, codon_predictions, all_solutions):
		if site == 1:
				stem = ""

				for code in bases:
					
					for residue_codons in codon_predictions:
						residue_possible = False
						residue_snippets = [e[:site] for e in residue_codons]
						
						for b in bases[code]:
							if stem + b in residue_snippets:
								residue_possible = True
								break
						if not residue_possible:
							break
					else:
						self._scs_helper(codon + code, site + 1, codon_predictions, all_solutions)

				return
			
		
		if site == 2:
			
			for code in bases:
				
				for residue_codons in codon_predictions:
					residue_possible = False
					residue_snippets = [e[:site] for e in residue_codons]
					for b in bases[code]:
						for stem in bases[codon]:
							if stem + b in residue_snippets:

								residue_possible = True
								continue
					if not residue_possible:
						break
				else:

					self._scs_helper(codon + code, site + 1, codon_predictions, all_solutions)

		if site == 3:

			for code in bases:
				
				for residue_codons in codon_predictions:
					residue_possible = False
					residue_snippets = [e[:site] for e in residue_codons]

					for b in bases[code]:
						for b1 in bases[codon[0]]:
							for b2 in bases[codon[1]]:
								stem = b1 + b2
								if stem + b in residue_snippets:
									residue_possible = True
									break
					if not residue_possible:
						break
				else: 
					self._scs_helper(codon + code, site + 1, codon_predictions, all_solutions)

			return

		if site == 4:
			all_solutions.append(codon)
			return


	def determine_best_codon(self, aa_predictions: List[str]):
		codon_predictions= [BT[e] for e in aa_predictions]

		max_eff = 0
		best_codon = None
		for solution in self._solve_codon_scheme(codon_predictions):
			possible_phenotypes = aas_from_codon(solution)
			possible_codons = codons_from_codon(solution)

			genotypes = len(possible_codons)

			phenotypes = len(possible_phenotypes)
			eff = phenotypes/genotypes

			if eff > max_eff:
				max_eff = eff
				best_codon = solution

		self.best_codon = best_codon
		self.efficiency = max_eff
		self.size = len(codons_from_codon(self.best_codon))
		return best_codon

	def _get_point_mutation(self, rank: int):
		return self.predictions[rank][0]



class PredictionCluster():
	def __init__(self, residues):
		self.residues = residues
		self.log_score = 0
		self.size = 1
		for res in self.residues:
			self.size *= res.size 
			self.log_score += res.score


	def __str__(self):
		residue_out = " ".join(f"{(str(res))}({res.best_codon})" for res in self.residues)
		return residue_out

	def print_point_mutations(self):
		residue_out = " ".join(f"{(str(res))}({res.point_mutation})" for res in self.residues)
		return residue_out

	def compute_efficiency(self):
		effs = [res.efficiency for res in self.residues]
		return sum(effs)/len(effs)

	def score(self):
		return len(self)*self.compute_efficiency()*self.log_score

	def __iter__(self):
		return iter(self.residues)

	def __len__(self):
		return len(self.residues)

	def __getitem__(self, key):
		return self.residues[key]

	def __eq__(self, other):
		if sorted(self.residues) == sorted(other.residues):
			return True
		return False


class Alignment():
	def __init__(self, nnoutput: NNOutput, gene_seq: str, chosen_chain_id: str):
		self.nnoutput = nnoutput
		self.gene_seq = gene_seq
		self.gene_aa_seq = "".join([FT[self.gene_seq[i*3:i*3+3]] for i in range(len(self.gene_seq) // 3)])
		self.chosen_chain_id = chosen_chain_id
		self.nn_start_position = None
		self.gene_start_position = None
		self.nn_aa_seq = None
		self.alignment = None
		self._prepare_alignment()


	def _prepare_alignment(self):
		self.nn_aa_seq = ""
		for idx, pred in enumerate(self.nnoutput):
			if idx == 0:
				self.nn_start_position = pred.position
			if pred.chain_id != self.chosen_chain_id:
				continue
			
			self.nn_aa_seq += pred.wtAA

		self.alignment = pairwise2.align.globalxs(self.nn_aa_seq, self.gene_aa_seq, -0.5, -0.1, one_alignment_only = True)[0]

		count = 0
		if self.alignment.seqA[0] == "-":
			while self.alignment.seqA[count] == "-":
		 		count += 1
		else:
		 	while self.alignment.seqB[-count] == "-":
		 		count -= 1

		self.gene_start_position = self.nn_start_position - count

	def idx(self, nn_bp_position: int):
		nn_aa_position = nn_bp_position // 3
		gene_aa_position = self.aa_idx(nn_aa_position)
		return gene_aa_position*3 + (nn_bp_position % 3)

	def aa_idx(self, nn_aa_position: int):
		return (nn_aa_position - (self.gene_start_position - 1))

	
	def __getitem__(self, nn_bp_position):
		if isinstance(nn_bp_position, slice):
			new_slice = slice(self.idx(nn_bp_position.start), self.idx(nn_bp_position.stop), nn_bp_position.step)
			return self.gene_seq[new_slice]
		else:
			return self.gene_seq[self.idx(nn_bp_position)]

	"""
	def __getitem__(self, nn_aa_position: int):
		if isinstance(nn_aa_position, slice):
			new_slice = slice(self.aa_idx(nn_aa_position.start), self.aa_idx(nn_aa_position.stop), nn_aa_position.step)
			print(new_slice)
			print(self.gene_aa_seq[new_slice])
			return self.gene_aa_seq[new_slice]
		else:
			return self.gene_aa_seq[self.aa_idx(nn_aa_position)]
	"""

class Primer(ABC):
	def __init__(self, forward: bool, aln: Alignment, cluster: PredictionCluster, ideal_tms: List[float], bp_range: List[int], tm_tolerance: float):
		self.forward = forward
		self.aln = aln
		self.cluster = cluster
		self.ideal_tms = ideal_tms
		self.bp_range = bp_range
		self.tm_tolerance = tm_tolerance
		self.regions = []
		self.forward = None

	def __str__(self):
		return "".join(self.regions)

	def __len__(self):
		return len(str(self))

	def determine_tm(self, region: int = 0) -> float:
		assert self.regions[region]
		sequence = self.regions[region]
		melting_temp = Tm_NN(sequence, Na = Na_eq)
		return melting_temp

	def total_tm(self):
		sequence = "".join(self.regions)
		melting_temp = Tm_NN(sequence, Na = Na_eq)
		
		return melting_temp 

	@abstractmethod
	def generate_primer(self, **kwargs):
		pass

class DirectedEvolutionForwardPrimer(Primer):
	def __init__(self, forward: bool, aln: Alignment, cluster: PredictionCluster, ideal_tms: List[float], bp_range: List[int], tm_tolerance: float):
		super().__init__(forward, aln, cluster, ideal_tms, bp_range, tm_tolerance)
		self.regions = ["", "", ""]
		self.fail = False

	def generate_primer(self, design_start: int, design_end: int):
		self.generate_homology(design_start)
		self.generate_annealing(design_end)
		self.generate_design(design_start, design_end)

	def generate_homology(self, design_start: int):
		for size in range(*self.bp_range):
			self.regions[0] = self.aln[design_start-size:design_start]
			left_tm = self.determine_tm(0)
			difference = self.ideal_tms[0] - left_tm
			if difference < self.tm_tolerance:
				break
		else:
			print("--Increasing # of homology BPs to achieve desired Tm")
			for size in range(self.bp_range[-1], self.bp_range[-1] + 15):
				self.regions[0] = self.aln[design_start-size:design_start]
				left_tm = self.determine_tm(0)
				difference = self.ideal_tms[0] - left_tm
				if difference < self.tm_tolerance:
					break
			else:
				self.fail = True


	def generate_annealing(self, design_end: int):
		for size in range(*self.bp_range):
			self.regions[-1] = self.aln[design_end+1:design_end+size]
			right_tm = self.determine_tm(-1)
			difference = self.ideal_tms[-1] - right_tm
			if difference < self.tm_tolerance and self.regions[-1][-1] in ["G", "C"]:
				break
		else:
			print("--Increasing # of annealing BPs to achieve desired Tm")
			for size in range(self.bp_range[-1], self.bp_range[-1] + 15):
				self.regions[-1] = self.aln[design_end+1:design_end+size]
				right_tm = self.determine_tm(-1)
				difference = self.ideal_tms[-1] - right_tm
				if difference < self.tm_tolerance and self.regions[-1][-1] in ["G", "C"]:
					break
			else:
				self.fail = True


	def generate_design(self, design_start: int, design_end: int):
		design = list(self.aln[design_start: design_end + 1])
		base_position = self.cluster[0].position

		for res in self.cluster:
			design[(res.position - base_position)*3:(res.position - base_position)*3+3] = res.best_codon

		design = "".join(design)	

		self.regions[1] = design


class DirectedEvolutionReversePrimer(Primer):
	def __init__(self, forward: bool, aln: Alignment, cluster: PredictionCluster, ideal_tms: List[float], bp_range: List[int], tm_tolerance: float):
		super().__init__(forward, aln, cluster, ideal_tms, bp_range, tm_tolerance)
		self.regions = ["", ""]
		self.fail = False	
	

	def generate_primer(self,homology_region: str, homology_region_start: int):
		self.generate_homology(homology_region)
		self.generate_annealing(homology_region_start)
		self.regions = [reverse_compliment(region) for region in self.regions]

	def generate_annealing(self, homology_region_start):
		for size in range(*self.bp_range):
			self.regions[-1] = self.aln[homology_region_start-(size+1):homology_region_start]
			tm = self.total_tm()
			difference = self.ideal_tms[0] - tm

			if difference < self.tm_tolerance and self.regions[-1][0] in ["G", "C"]:
				break

		
		else:
			print("--Increasing # of reverse annealing BPs to achieve desired Tm")
			for size in range(self.bp_range[-1], self.bp_range[-1] + 15):
				self.regions[-1] = self.aln[homology_region_start-(size+1):homology_region_start]
				tm = self.total_tm()
				difference = self.ideal_tms[0] - tm

				if difference < self.tm_tolerance and self.regions[-1][0] in ["G", "C"]:
					break


			else:
				self.fail = True		


	def generate_homology(self, homology_region: str):	
		self.regions[0] = homology_region

"""
class DirectedEvolutionReversePrimer(Primer):
	def __init__(self, forward: bool, aln: Alignment, cluster: PredictionCluster, ideal_tms: List[float], bp_range: List[int], tm_tolerance: float):
		super().__init__(forward, aln, cluster, ideal_tms, bp_range, tm_tolerance)
		self.regions = [""]
		self.fail = False

	def generate_primer(self, homology_region: str, homology_region_start: int):
		reverse_primer = reverse_compliment(homology_region)
		additional_bps = 15
		for i in range(additional_bps):
			self.regions[0] +=  self.aln[homology_region_start - (i+1)]
			primer_tm = self.determine_tm(0)
			difference = self.ideal_tms[0] - primer_tm

			if (difference < self.tm_tolerance) & (reverse_primer[-1] in ["G", "C"]):
				break
		else:
			print("--Increasing size of reverse primer to achieve desired Tm")
			for i in range(additional_bps, 71 - len(reverse_primer)):
				self.regions[0] +=  self.aln[homology_region_start - (i+1)]
				primer_tm = self.determine_tm(0)
				difference = self.ideal_tms[0] - primer_tm

				if (difference < self.tm_tolerance) & (reverse_primer[-1] in ["G", "C"]):
					break
			else:
				fail = True

"""

class SDMForwardPrimer(Primer):
	def __init__(self, forward: bool, aln: Alignment, cluster: PredictionCluster, ideal_tms: List[float], bp_range: List[int], tm_tolerance: float):
		super().__init__(forward, aln, cluster, ideal_tms, bp_range, tm_tolerance)
		self.regions = ["", "", ""]
		self.fail = False	

	def generate_primer(self, design_start: int, design_end: int):
		self.generate_design(design_start, design_end)
		self.generate_annealing(design_start, design_end)

	def generate_annealing(self, design_start: int, design_end: int):
		for size in range(*self.bp_range):
			self.regions[0] = self.aln[design_start-size//2:design_start]
			tm = self.total_tm()
			difference = self.ideal_tms[0] - tm

			if difference < self.tm_tolerance and self.regions[-1][-1] in ["G", "C"]:
				break

			self.regions[-1] = self.aln[design_end+1:design_end+size]
			tm = self.total_tm()
			difference = self.ideal_tms[0] - tm

			if difference < self.tm_tolerance and self.regions[-1][-1] in ["G", "C"]:
				break

			self.regions[-1] = self.aln[design_end+1:design_end+size+1]
			tm = self.total_tm()
			difference = self.ideal_tms[0] - tm

			if difference < self.tm_tolerance and self.regions[-1][-1] in ["G", "C"]:
				break
		
		else:
			print("--Increasing # of forward annealing BPs to achieve desired Tm")
			for size in range(self.bp_range[-1], self.bp_range[-1] + 15):
				self.regions[0] = self.aln[design_start-size//2:design_start]
				tm = self.total_tm()
				difference = self.ideal_tms[0] - tm

				if difference < self.tm_tolerance and self.regions[-1][-1] in ["G", "C"]:
					break

				self.regions[-1] = self.aln[design_end+1:design_end+size]
				tm = self.total_tm()
				difference = self.ideal_tms[0] - tm

				if difference < self.tm_tolerance and self.regions[-1][-1] in ["G", "C"]:
					break
				
				self.regions[-1] = self.aln[design_end+1:design_end+size+1]
				tm = self.total_tm()
				difference = self.ideal_tms[0] - tm

				if difference < self.tm_tolerance and self.regions[-1][-1] in ["G", "C"]:
					break
								

			else:
				self.fail = True		

	def generate_design(self, design_start: int, design_end: int):
		design = list(self.aln[design_start: design_end + 1])
		base_position = self.cluster[0].position

		for res in self.cluster:
			design[(res.position - base_position)*3:(res.position - base_position)*3+3] = BT[res.point_mutation][0]

		design = "".join(design)	

		self.regions[1] = design

class SDMReversePrimer(Primer):
	def __init__(self, forward: bool, aln: Alignment, cluster: PredictionCluster, ideal_tms: List[float], bp_range: List[int], tm_tolerance: float):
		super().__init__(forward, aln, cluster, ideal_tms, bp_range, tm_tolerance)
		self.regions = ["", "", ""]
		self.fail = False	
	

	def generate_primer(self, design_start: int, design_end: int):
		self.generate_design(design_start, design_end)
		self.generate_annealing(design_start, design_end)
		self.regions = [reverse_compliment(region) for region in self.regions]

	def generate_annealing(self, design_start: int, design_end: int):
		for size in range(*self.bp_range):
			self.regions[0] = self.aln[design_end+1:design_end+size//2]
			tm = self.total_tm()
			difference = self.ideal_tms[0] - tm

			if difference < self.tm_tolerance and self.regions[-1][0] in ["G", "C"]:
				break
			self.regions[-1] = self.aln[design_start-size:design_start]
			tm = self.total_tm()
			difference = self.ideal_tms[0] - tm

			if difference < self.tm_tolerance and self.regions[-1][0] in ["G", "C"]:
				break

			self.regions[-1] = self.aln[design_start-(size+1):design_start]
			tm = self.total_tm()
			difference = self.ideal_tms[0] - tm

			if difference < self.tm_tolerance and self.regions[-1][0] in ["G", "C"]:
				break

		
		else:
			print("--Increasing # of reverse annealing BPs to achieve desired Tm")
			for size in range(self.bp_range[-1], self.bp_range[-1] + 15):
				self.regions[0] = self.aln[design_end+1:design_end+size//2]
				tm = self.total_tm()
				difference = self.ideal_tms[0] - tm

				if difference < self.tm_tolerance and self.regions[-1][0] in ["G", "C"]:
					break

				self.regions[-1] = self.aln[design_start-size:design_start]
				tm = self.total_tm()
				difference = self.ideal_tms[0] - tm

				if difference < self.tm_tolerance and self.regions[-1][0] in ["G", "C"]:
					break

				self.regions[-1] = self.aln[design_start-(size+1):design_start]
				tm = self.total_tm()
				difference = self.ideal_tms[0] - tm

				if difference < self.tm_tolerance and self.regions[-1][0] in ["G", "C"]:
					break


			else:
				self.fail = True		


	def generate_design(self, design_start: int, design_end: int):
		design = list(self.aln[design_start: design_end + 1])
		base_position = self.cluster[0].position

		for res in self.cluster:
			design[(res.position - base_position)*3:(res.position - base_position)*3+3] = BT[res.point_mutation][0]

		design = "".join(design)	

		self.regions[1] = design


class PrimerGenerator(ABC):
	def __init__(self, aln: Alignment, ideal_tms: List[float], bp_range: List[float], tm_tolerance: float):
		self.aln = aln
		self.ideal_tms = ideal_tms
		self.tm_tolerance = tm_tolerance
		self.bp_range = bp_range

	@abstractmethod
	def generate_primers(self, cluster: PredictionCluster):
		pass


class DirectedEvolutionPrimerGenerator(PrimerGenerator):
	def generate_primers(self, cluster: PredictionCluster):
		design_start = cluster[0].position * 3 - 3
		design_end = cluster[-1].position * 3 - 1

		forward = DirectedEvolutionForwardPrimer(True, self.aln, cluster, self.ideal_tms[0:2], self.bp_range, self.tm_tolerance)
		forward.generate_primer(design_start, design_end)

		if forward.fail:
			print("Cannot meet Tm requirements for homology or annealing")
			return

		reverse = DirectedEvolutionReversePrimer(False, self.aln, cluster, [self.ideal_tms[2]], [0], self.tm_tolerance)
		reverse.generate_primer(forward.regions[0], cluster[0].position*3-3-len(forward.regions[0]))
		
		if reverse.fail:
			print("Cannot meet Tm requirements for reverse primer")
			return

		return forward, reverse

class SDMPrimerGenerator(PrimerGenerator):
	def generate_primers(self, cluster: PredictionCluster):
		design_start = cluster[0].position * 3 - 3
		design_end = cluster[-1].position * 3 - 1

		forward = SDMForwardPrimer(True, self.aln, cluster, [self.ideal_tms[0]], self.bp_range, self.tm_tolerance)
		forward.generate_primer(design_start, design_end)

		if forward.fail:
			print("Cannot meet Tm requirements for homology or annealing")
			return

		reverse = SDMReversePrimer(False, self.aln, cluster, [self.ideal_tms[1]], self.bp_range, self.tm_tolerance)
		reverse.generate_primer(design_start, design_end)
		
		if reverse.fail:
			print("Cannot meet Tm requirements for reverse primer")
			return

		return forward, reverse

class ClusterGenerator(ABC):
	def __init__(self, mutagenesis_aa_length: int):
		self.mutagenesis_aa_length = mutagenesis_aa_length

	def predictions_to_clusters(self, mutable_residues: List[Prediction]) -> List[PredictionCluster]:
		clusters = []
		for i, base_pred in enumerate(mutable_residues[:-1]):
			cluster = [base_pred]

			for j, pred in enumerate(mutable_residues[i+1:]):
				if (pred.position - base_pred.position < self.mutagenesis_aa_length) and (pred.chain_id == pred.chain_id):
					cluster.append(pred)
				else:
					break

			cluster = PredictionCluster(cluster)
			if self.cluster_requirements(cluster):
				clusters.append(cluster)

		clusters = sorted(clusters, key = lambda x: x.score(), reverse = True)	
		return clusters


	@abstractmethod
	def cluster_requirements(self, cluster: PredictionCluster) -> bool:
		pass


class DirectedEvolutionClusterGenerator(ClusterGenerator):
	def __init__(self, mutagenesis_aa_length: int, library_size: int, size_tolerance: float):
		super().__init__(mutagenesis_aa_length)
		self.library_size = library_size
		self.size_tolerance = size_tolerance

	def cluster_requirements(self, cluster: PredictionCluster) -> bool:
		if abs(math.log(cluster.size/self.library_size, 10)) > self.size_tolerance:
			return False
		else:
			return True

class SDMClusterGenerator(ClusterGenerator):
	def __init__(self, mutagenesis_aa_length: int, num_mutations: int):
		
		super().__init__(mutagenesis_aa_length)
		self.num_mutations = num_mutations

	def cluster_requirements(self, cluster: PredictionCluster) -> bool:
		if len(cluster) == self.num_mutations:
			return True
		else:
			return False



class PointMutation():
	def __init__(self, chain_id, position, label, point_mutation):
		self.chain_id = chain_id
		self.position = position
		self.label = label
		self.point_mutation = point_mutation

class PointMutationReader():
	def __init__(self, chain_id: str):
		self.chain_id = chain_id

	def read_mutations(self, point_mutation_file: Path) -> List[List[PointMutation]]:
		point_mutations = []
		with open(point_mutation_file) as f:
			for line in f:
				point_mutation_strings = [e.strip() for e in line.split(" ")]
				point_mutation_cluster = [PointMutation(self.chain_id, int(e[1:-1]), e[0], e[-1]) for e in point_mutation_strings]
				point_mutations.append(point_mutation_cluster)	

		return point_mutations



class PointMutationToClusters():
	def __init__(self, nnoutput: NNOutput):
		self.nnoutput = nnoutput

	def point_mutations_to_clusters(self, point_mutations: List[List[PointMutation]]):
		clusters = []
		for point_mutation_cluster in point_mutations:
			cluster = []
			for point_mutation in point_mutation_cluster:
				pred = self.nnoutput.get_prediction(point_mutation.chain_id, point_mutation.position)
				pred.determine_best_codon([point_mutation.label])
				pred.point_mutation = point_mutation.point_mutation
				cluster.append(pred)

			cluster = PredictionCluster(cluster)
			clusters.append(cluster)

		return clusters


def generate_point_mutation_primers(sequence: str, prediction: Path, point_mutation_file: Path, output: Path, chosen_chain_id: str, mutagenesis_length: int, ideal_tms: List[float], tm_tolerance: float, na_eq: float):
	global Na_eq
	Na_eq = na_eq
	nnoutput = NNOutput(prediction)
	pmr = PointMutationReader(chosen_chain_id)
	point_mutations = pmr.read_mutations(point_mutation_file)
	pm_cg = PointMutationToClusters(nnoutput)
	clusters = pm_cg.point_mutations_to_clusters(point_mutations)

	min_homology_len = 1
	max_homology_len = 30
	bp_range = [min_homology_len, max_homology_len]

	aln = Alignment(nnoutput, sequence, chosen_chain_id)

	all_clusters = []

	with open(output, "w+") as log:
		
		log.write("Score, Mutations, # Mutation Sites, Size, Forward Tm, Reverse Tm, Forward Primer Length, Reverse Primer Length, Forward Primer, Reverse Primer,\n")
	
		for cluster in clusters[:]:
			if cluster in all_clusters:
				print(f"Ignoring duplicate cluster")
				continue
			
			all_clusters.append(cluster)

			sdm_pg = SDMPrimerGenerator(aln, ideal_tms, bp_range, tm_tolerance)
			primers = sdm_pg.generate_primers(cluster)

			if primers is None:
				continue

			forward, reverse = primers

			log.write(",".join([str(cluster.score()), str(cluster.print_point_mutations()), str(len(cluster)), str(cluster.size), 
								str(forward.total_tm()),  str(reverse.total_tm()), 
								str(len(forward)), str(len(reverse)), str(forward), str(reverse),"\n"]))		



def generate_ssm_primers(sequence: str, prediction: Path, output: Path, chosen_chain_id: str , library_size: int, size_tolerance: float, mutagenesis_length: int, ideal_tms: List[float], tm_tolerance: float, na_eq: float):
	# Open NNOutput
	global Na_eq
	Na_eq = na_eq


	aa_window_size = mutagenesis_length//3
	min_homology_len = 18
	max_homology_len = 30
	bp_range = [min_homology_len, max_homology_len]
	nnoutput = NNOutput(prediction)

	aln = Alignment(nnoutput, sequence, chosen_chain_id)
	print("Beginning to generate mixed base clusters")
	all_clusters = []

	with open(output, "w+") as log:
		
		log.write("Score, Misprediction Cutoff, Cumalative Probability Cutoff, Mutations, # Mutation Sites, Size, Homology Tm, Annealing Tm, Reverse Tm, Forward Primer Length, Reverse Primer Length, Forward Primer, Reverse Primer,\n")
		
		for cum_prob_cutoff in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
			for misprediction_cutoff in [0.1, 0.15, 0.20, 0.25, 0.3, 0.350, 0.40, 0.450, 0.50]:
				
				print(f"\nAttempting to generate clusters at a misprediction cutoff of {misprediction_cutoff} and a cumaltive probability cutoff of {cum_prob_cutoff}")
				
				mutable_residues = nnoutput.determine_mutable_residues(chosen_chain_id, misprediction_cutoff, cum_prob_cutoff)

				de_cg = DirectedEvolutionClusterGenerator(aa_window_size, library_size, size_tolerance)
				clusters = de_cg.predictions_to_clusters(mutable_residues)
				
				for cluster in clusters[:5]:
					if cluster in all_clusters:
						print(f"Ignoring duplicate cluster")
						continue
					
					print(f"Successfully found cluster of size {cluster.size}")
					all_clusters.append(cluster)

					de_pg = DirectedEvolutionPrimerGenerator(aln, ideal_tms, bp_range, tm_tolerance)
					primers = de_pg.generate_primers(cluster)

					if primers is None:
						continue

					forward, reverse = primers

					log.write(",".join([str(cluster.score()), str(misprediction_cutoff), str(cum_prob_cutoff), str(cluster), str(len(cluster)), str(cluster.size), 
										str(forward.determine_tm(0)), str(forward.determine_tm(-1)), str(reverse.determine_tm(0)), 
										str(len(forward)), str(len(reverse)), str(forward), str(reverse),"\n"]))		



def generate_sdm_primers(sequence: str, prediction: Path, output: Path, chosen_chain_id: str , num_mutations: int, mutagenesis_length: int, ideal_tms: List[float], tm_tolerance: float, na_eq: float):
	global Na_eq
	Na_eq = na_eq

	aa_window_size = mutagenesis_length//3
	min_homology_len = 1
	max_homology_len = 20
	bp_range = [min_homology_len, max_homology_len]
	nnoutput = NNOutput(prediction)

	aln = Alignment(nnoutput, sequence, chosen_chain_id)
	print("Beginning to generate sdm clusters")
	all_clusters = []

	with open(output, "w+") as log:
		
		log.write("Score, Misprediction Cutoff, Cumalative Probability Cutoff, Mutations, # Mutation Sites, Size, Forward Tm, Reverse Tm, Forward Primer Length, Reverse Primer Length, Forward Primer, Reverse Primer,\n")
		
		for cum_prob_cutoff in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
			for misprediction_cutoff in [0.1, 0.15, 0.20, 0.25, 0.3, 0.350, 0.40, 0.450, 0.50]:
				
				print(f"\nAttempting to generate clusters at a misprediction cutoff of {misprediction_cutoff} and a cumaltive probability cutoff of {cum_prob_cutoff}")
				
				mutable_residues = nnoutput.determine_mutable_residues(chosen_chain_id, misprediction_cutoff, cum_prob_cutoff)

				pm_cg = SDMClusterGenerator(aa_window_size, num_mutations)
				clusters = pm_cg.predictions_to_clusters(mutable_residues)
				
				for cluster in clusters[:5]:
					if cluster in all_clusters:
						print(f"Ignoring duplicate cluster")
						continue
					
					print(f"Successfully found cluster of size {cluster.size}")
					all_clusters.append(cluster)

					sdm_pg = SDMPrimerGenerator(aln, ideal_tms, bp_range, tm_tolerance)
					primers = sdm_pg.generate_primers(cluster)

					if primers is None:
						continue

					forward, reverse = primers

					log.write(",".join([str(cluster.score()), str(misprediction_cutoff), str(cum_prob_cutoff), str(cluster.print_point_mutations()), str(len(cluster)), str(cluster.size), 
										str(forward.total_tm()),  str(reverse.total_tm()), 
										str(len(forward)), str(len(reverse)), str(forward), str(reverse),"\n"]))		






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
						default = 65.0)
	parser.add_argument("-rtm", "--reverse_tm",
						help = "Melting temperature for reverse primer",
						type = float,
						default = 68.0)
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
	generate_ssm_primers(args.sequence, args.prediction, args.output, args.chain_id, args.library_size, args.size_tolerance, args.mutagenesis_length, [args.homology_tm, args.annealing_tm, args.reverse_tm], args.tm_tolerance, args.na_eq)


if __name__ == "__main__":
	cli(sys.argv[1:])