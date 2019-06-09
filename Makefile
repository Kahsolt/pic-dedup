rundev:
	python dedup.py

run: dist
	python dedup_standalone.py

dist:
	python -m 'dedup.distro'

clean:
	rm -rf dedup/__pycache__/
	rm -f dedup_standalone.py

backup:
	cp index.db index.bak

sql:
	sqlite3 index.db

stat:
	@sqlite3 index.db "SELECT name, COUNT(*) FROM Picture JOIN Folder ON Picture.folder_id = Folder.id GROUP BY name;"
