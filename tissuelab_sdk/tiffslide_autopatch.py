import sys
import re
from pathlib import Path


def _auto_patch_tiffslide() -> bool:
    """
    Automatically patch tiffslide to support 4D z-stack at runtime.
    
    Returns:
        bool: True if patches are applied or already present
    """
    try:
        import tiffslide
        print(f"[TissueLab-SDK] Checking tiffslide {tiffslide.__version__}...")
    except ImportError:
        # tiffslide not installed, skip silently
        return True
    
    try:
        tiffslide_dir = Path(tiffslide.__file__).parent
        target_file = tiffslide_dir / 'tiffslide.py'
        
        if not target_file.exists():
            return True
        
        # Read current content
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            with open(target_file, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Check if already patched
        if re.search(r'series\.ndim not in \([^)]*4[^)]*\)', content):
            print(f"[TissueLab-SDK] TiffSlide {tiffslide.__version__} already patched, skipping")
            return True  # Already supports 4D
        
        # Create backup
        backup_file = target_file.with_suffix('.py.tissuelab_backup')
        if not backup_file.exists():
            import shutil
            shutil.copy2(target_file, backup_file)
        
        original_content = content
        
        # Patch 1: Support 4D series
        pattern1_old = r'(if\s+series\.ndim\s*!=\s*3\s*:)'
        pattern1_new = r'(if\s+series\.ndim\s+not\s+in\s+\(2,\s*3\)\s*:)'
        
        if re.search(pattern1_old, content):
            content = re.sub(pattern1_old, r'if series.ndim not in (3, 4):', content)
        elif re.search(pattern1_new, content):
            content = re.sub(pattern1_new, r'if series.ndim not in (2, 3, 4):', content)
        
        # Patch 2: ZYXS axes support
        if 'elif axes == "ZYXS":' not in content:
            pattern2 = r'([\s]*)(else\s*:\s*\n\s*raise\s+NotImplementedError\(f?["\']series with axes=)'
            match = re.search(pattern2, content)
            if match:
                indent = match.group(1)
                zyxs_block = f'''{indent}elif axes == "ZYXS":
{indent}    _, h0, w0, _ = map(int, series.shape)
{indent}    level_dimensions = ((lvl.shape[2], lvl.shape[1]) for lvl in series.levels)
{indent}'''
                content = re.sub(pattern2, zyxs_block + r'\2', content, count=1)
        
        # Patch 3: ZYXS tile size
        if not ('if axes == "ZYXS":' in content and '.pages[0]' in content):
            pattern3 = r'(\s+)(page\s*=\s*series\.levels\[lvl\]\[0\])'
            if re.search(pattern3, content):
                def replace_page(match):
                    indent = match.group(1)
                    return f'''{indent}if axes == "ZYXS":
{indent}    page = series.levels[lvl].pages[0]
{indent}else:
{indent}    page = series.levels[lvl][0]'''
                content = re.sub(pattern3, replace_page, content, count=1)
        
        # Patch 4: ZYXS read_region
        has_sel = re.search(r'elif\s+axes\s*==\s*["\']ZYXS["\']\s*:.*?selection\s*=\s*slice\(0,\s*1\)', content, re.DOTALL)
        has_trans = re.search(r'elif\s+axes\s*==\s*["\']ZYXS["\']\s*:.*?arr\s*=\s*arr\[0,\s*:,\s*:,\s*:\]', content, re.DOTALL)
        
        if not has_sel:
            pattern4a = r'([\s]*)(else\s*:\s*\n\s*raise\s+NotImplementedError\(f?["\']axes=)'
            match = re.search(pattern4a, content)
            if match:
                indent = match.group(1)
                zyxs_sel = f'''{indent}elif axes == "ZYXS":
{indent}    selection = slice(0, 1), slice(ry0, ry1), slice(rx0, rx1), slice(None)
{indent}'''
                content = re.sub(pattern4a, zyxs_sel + r'\2', content, count=1)
        
        if not has_trans:
            pattern4b = r'(if\s+axes\s*==\s*["\']CYX["\']\s*:\s*\n\s+arr\s*=\s*arr\.transpose\(.*?\)\n)'
            match = re.search(pattern4b, content)
            if match:
                indent_match = re.search(r'(\s+)arr\s*=', match.group(0))
                indent = indent_match.group(1) if indent_match else '            '
                zyxs_trans = f'''{indent[:-4]}elif axes == "ZYXS":
{indent}arr = arr[0, :, :, :]
'''
                content = re.sub(pattern4b, r'\1' + zyxs_trans, content, count=1)
        
        # Write back if changed
        if content != original_content:
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[TissueLab-SDK] Auto-patched tiffslide {tiffslide.__version__} for 4D z-stack support")
            
            # CRITICAL: Reload tiffslide module to pick up the changes
            # If tiffslide was already imported, we need to reload it
            if 'tiffslide' in sys.modules:
                import importlib
                importlib.reload(sys.modules['tiffslide'])
                print(f"[TissueLab-SDK] Reloaded tiffslide module")
            
            return True
        
        return True
            
    except Exception as e:
        # Silently fail - don't break imports
        import warnings
        warnings.warn(f"TissueLab-SDK: Failed to auto-patch tiffslide: {e}")
        return False


# Auto-execute when this module is imported
_patched = _auto_patch_tiffslide()

