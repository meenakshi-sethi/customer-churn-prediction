from pathlib import Path

from app import app


project_root = Path(__file__).resolve().parents[1]
app.config['STATIC_REPORT_EXPORT'] = True

with app.test_request_context('/report'):
    html = app.view_functions['report']()
html = '\n'.join(line.rstrip() for line in html.splitlines()) + '\n'

output_path = project_root / 'docs' / 'report.html'
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(html, encoding='utf-8')
print(f'Wrote static report to {output_path.relative_to(project_root)}')
