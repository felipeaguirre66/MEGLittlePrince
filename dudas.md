1. **¿La señal del *_meg.fif ya está band-pass filtrada? ¿A qué frecuencias?**

	Sí tiene filtrado aplicado durante la adquisición: el sistema usó un pasa-altos
	online de **0.1 Hz** y un pasa-bajos online de **330 Hz**. La frecuencia de
	muestreo es 1000 Hz. No encontré evidencia de un band-pass digital offline
	aplicado como preprocesamiento. Ojo: el channels.tsv informa límites de
	0–330 Hz y el *_meg.json no detalla un filtro digital; para el filtro de
	adquisición, la documentación del dataset especifica 0.1–330 Hz. Véase
	[MEG_README.md](MEG_README.md) y el [notebook de inspección de un run](nbs/%5B2026-09-30%5D%20run%20first%20dive.ipynb).

2. **¿Ya están identificados los “noisy and flat channels”? ¿Dónde?**

	No hay canales marcados como malos en los metadatos BIDS: revisé los
	*_channels.tsv de todos los sujetos y runs y todos tienen status=good;
	no aparece una lista de canales ruidosos o planos. Esto significa que no están
	identificados allí, no que se haya demostrado que todos sean buenos: haría
	falta control de calidad (p. ej., inspección de señal/PSD) para detectarlos.

3. **¿El events.tsv ya está alineado al *_meg.fif?**

	No asumiría que sí tal como está; al menos el run 01 de `sub-01` muestra una
	discrepancia. El FIF dura **622.999 s**, pero el events.tsv va desde onset
	**90.175 s** hasta aproximadamente **697.499 s** (fin del último evento), es
	decir, algunos eventos quedan fuera del archivo si `onset` se interpreta desde
	el comienzo del FIF. En sourcedata/task-listen_run-01_extra_info.tsv, los
	eventos correspondientes van de **3.05 s** a **610.38 s**: entre ambas series
	hay un desplazamiento de aproximadamente **87.12 s**. Antes de usar los eventos
	para epoching, hay que aclarar/corregir ese offset y validarlo contra los
	triggers o el inicio real del audio. El cálculo exploratorio está en el
	[notebook de inspección de un run](nbs/%5B2026-09-30%5D%20run%20first%20dive.ipynb).