# SOP-QC-08: Standard Operating Procedure for Defect Triage

## 1. Purpose & Scope
This SOP defines the visual inspection, classification, and routing procedure for manufacturing defects identified on structural brackets, metal castings, and assemblies.

## 2. Defect Classification Scheme
Defects must be classified into one of the following severity levels upon detection:

### A. Critical Defects (Refusal / Line Stop)
- **Definition**: Any defect that compromises structural integrity, safety, or basic functionality.
- **Examples**: Through-thickness cracks, open shrinkage cavities in load-bearing nodes, severe casting porosity.
- **Action**: Immediate line stop. Route to Chief Metallurgist and QA Director. Quarantine the batch.

### B. Major Defects (Rework Required)
- **Definition**: Non-critical anomalies that exceed tolerance limits but can be repaired via approved rework procedures.
- **Examples**: Surface cracks > 2mm, cold shuts, welding undercut > 1mm.
- **Action**: Segregate part. Route to Rework Station 3. Log in MLflow tracking runs.

### C. Minor Defects (Acceptable / Cosmetic)
- **Definition**: Minor cosmetic deviations that do not impact mechanical strength or product lifetime.
- **Examples**: Small surface scratches (< 1mm), minor paint discoloration.
- **Action**: Log in system and release for packaging.

## 3. Triage Report Requirements
Every triage entry must contain:
1. Part Serial ID (OCR extracted).
2. Inspection Photo reference (Image ID).
3. Defect Tag (from Casting/Welding Taxonomy).
4. Severity Classification.
