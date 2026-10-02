## Summary

This dataset contains magnetoencephalography (MEG) recordings collected
while participants listened to the French audiobook of *Le Petit Prince*
by Antoine de Saint-Exupéry.

A complementary MEG dataset from the same project, using a reading (RSVP) paradigm, is available on OpenNeuro (accession number: ds007524).

This data is analyzed in:

d’Ascoli, S., Bel, C., Rapin, J. et al. Towards decoding individual words from non-invasive brain recordings. Nature Communications 16, 10521 (2025). https://doi.org/10.1038/s41467-025-65499-0

------------------------------------------------------------------------

## Participants

Fifty-eight healthy adults participated in the listening experiment (17
females; mean age = 27.8 years, SD = 5.5 years).

All participants were native French speakers, right-handed, and reported
no history of neurological disorders. Written informed consent was
obtained prior to participation. The study was approved by the relevant
local ethics committee.

------------------------------------------------------------------------

## Stimuli

The auditory stimulus consisted of the French audiobook version of *Le
Petit Prince*.

- Language: French
- Format: Continuous audiobook
- Segmentation: 9 parts
- Mean duration per part: 10min50s
- Standard deviation: 55s
- Minimum duration: 9min40s
- Maximum duration: 12min30s

The same audiobook version was previously used in a publicly available
fMRI dataset (Li et al., 2022). 

------------------------------------------------------------------------

## Experimental Procedure

Participants were seated in the MEG system after informed consent and
familiarization with the recording environment.

Auditory stimuli were delivered through MEG-compatible earphones. Sound
intensity was individually adjusted to a comfortable listening level
before the experiment. Participants were instructed to listen
attentively and remain as still as possible.

The experiment consisted of 9 runs, corresponding to the 9 audiobook
segments. Between runs, participants completed 4 multiple-choice
comprehension questions presented visually on a screen (not reported here). 
Short breaks were provided between runs. Alertness and movement were monitored 
via camera during recording.

------------------------------------------------------------------------

## Acquisition

### MEG

MEG data for all three tasks were recorded inside the same magnetically shielded room using a whole-head Elekta Neuromag TRIUX MEG system (Elekta Oy, Helsinki, Finland), equipped with 102 magnetometers and 204 planar gradiometers. Data were recorded continuously with a sampling rate of 1000 Hz and an online low-pass filter at 330 Hz and high-pass filter at 0.1 Hz.
Vertical and horizontal electrooculograms (EOG) and an electrocardiogram (ECG) were recorded simultaneously using bipolar electrodes to monitor eye movements and heartbeats.

### Anatomical MRI

For each participant, a high-resolution T1-weighted anatomical MRI scan was acquired using a 3T Siemens Magnetom Prisma MRI scanner (Siemens Healthcare, Erlangen, Germany).
A standard MPRAGE sequence was used. MRI scans were typically acquired right after the MEG recording. Scans were used for coregistration and cortical surface reconstruction for source analysis. 

------------------------------------------------------------------------

## Data Organization

### Raw Data

The root directory includes:

-   `dataset_description.json`
-   `participants.tsv` and `participants.json`
-   `task-listen_events.json`
-   `sub-01` to `sub-58`
-   `sourcedata/`

Each subject directory (`sub-XX`) contains one session (`ses-01`) with:

-   `anat/`: T1-weighted MRI (`sub-XX_ses-01_T1w.nii.gz`) and
    corresponding JSON sidecar
-   `meg/`: 9 MEG runs (`task-listen_run-01` to `run-09`), each
    including:
    -   continuous MEG data (`*_meg.fif`)
    -   sidecar JSON files
    -   `events.tsv` and `channels.tsv` files
    -   coordinate system file (`*_coordsystem.json`)
    -   calibration and crosstalk files
-   `sub-XX_ses-01_scans.tsv`: scan-level metadata

Each run corresponds to one audiobook segment.

Acquisition parameters are provided in the corresponding sidecar JSON
files.

------------------------------------------------------------------------

## References

Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G.,
Gramfort, A., Henson, R. N., Jas, M., Litvak, V., Moreau, J.,
Oostenveld, R., Schoffelen, J., Tadel, F., Wexler, J., & Baillet, S.
(2018). MEG-BIDS, the brain imaging data structure extended to
magnetoencephalography. *Scientific Data*, 5, 180110.
https://doi.org/10.1038/sdata.2018.110


Li, Jixing, et al. “Le Petit Prince Multilingual Naturalistic fMRI Corpus.” Scientific Data, vol. 9, no. 1, Aug. 2022, p. 530. www.nature.com, https://doi.org/10.1038/s41597-022-01625-7.

