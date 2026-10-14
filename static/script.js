document.addEventListener('DOMContentLoaded', () => {
    const $ = id => document.getElementById(id);
    const dropzone = $('dropzone'), clothingDropzone = $('clothingDropzone'), fileInput = $('fileInput'), clothingFileInput = $('clothingFileInput'), fitButton = $('fitButton'), loader = $('loader'), message = $('formMessage'), gallery = $('historyGallery');
    let personFile = null, garmentFile = null, garmentUrl = null, galleryImages = [], lightboxIndex = 0;
    
    const extractGarmentBtn = $('extractGarmentBtn');
    const outfitFileInput = $('outfitFileInput');
    const extractLoader = $('extractLoader');
    const extractedGarments = $('extractedGarments');

    function setPreview(element, file, urlOverride = null) {
        const url = urlOverride || URL.createObjectURL(file);
        element.innerHTML = `<img src="${url}" alt="Selected upload"><span class="preview-label">${file.name}</span>`;
        element.classList.add('has-image');
    }
    
    function updateButton() {
        fitButton.disabled = !(personFile && (garmentFile || garmentUrl));
    }

    function choose(file, setter, previewId) {
        if (!file.type.startsWith('image/')) {
            message.textContent = 'Please choose an image file.';
            return;
        }
        setter(file);
        if (previewId === 'clothingPreview') garmentUrl = null;
        setPreview($(previewId), file);
        message.textContent = '';
        updateButton();
    }
    
    function bindUpload(zone, input, setter, previewId) {
        zone.addEventListener('click', event => {
            if (event.target.tagName !== 'BUTTON') input.click();
        });
        zone.addEventListener('dragover', event => {
            event.preventDefault();
            zone.classList.add('dragover');
        });
        zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
        zone.addEventListener('drop', event => {
            event.preventDefault();
            zone.classList.remove('dragover');
            if (event.dataTransfer.files[0]) choose(event.dataTransfer.files[0], setter, previewId);
        });
        input.addEventListener('change', () => {
            if (input.files[0]) choose(input.files[0], setter, previewId);
        });
    }
    
    $('browseBtn').addEventListener('click', event => { event.stopPropagation(); fileInput.click(); });
    $('clothingBrowseBtn').addEventListener('click', event => { event.stopPropagation(); clothingFileInput.click(); });
    
    bindUpload(dropzone, fileInput, file => { personFile = file; }, 'personPreview');
    bindUpload(clothingDropzone, clothingFileInput, file => { garmentFile = file; }, 'clothingPreview');
    
    extractGarmentBtn.addEventListener('click', () => outfitFileInput.click());
    outfitFileInput.addEventListener('change', () => {
        if (!outfitFileInput.files[0]) return;
        
        extractGarmentBtn.disabled = true;
        extractLoader.classList.remove('hidden');
        extractedGarments.classList.add('hidden');
        extractedGarments.innerHTML = '';
        
        const fd = new FormData();
        fd.append('file', outfitFileInput.files[0]);
        
        fetch('/extract_garments', {
            method: 'POST',
            body: fd
        })
        .then(res => res.json())
        .then(data => {
            if (!data.success) throw new Error(data.error || 'Failed to extract.');
            
            if (data.items.length === 0) {
                alert("No garments detected in the image.");
                return;
            }
            
            data.items.forEach(item => {
                addWardrobeItem(item, true);
                const img = document.createElement('img');
                img.src = item.url;
                img.className = 'extracted-item';
                img.style.width = '100%';
                img.style.aspectRatio = '3/4';
                img.style.objectFit = 'cover';
                img.style.cursor = 'pointer';
                img.style.border = '2px solid transparent';
                img.style.borderRadius = '4px';
                
                img.addEventListener('click', () => {
                    garmentFile = null;
                    garmentUrl = item.filename;
                    $('clothingPreview').innerHTML = `<img src="${item.url}" alt="Extracted garment" style="object-fit: cover; width: 100%; height: 100%;"><span class="preview-label">${item.type}</span>`;
                    $('clothingPreview').classList.add('has-image');
                    $('clothingTypeSelect').value = item.type;
                    updateButton();
                    
                    document.querySelectorAll('.extracted-item').forEach(el => el.style.borderColor = 'transparent');
                    img.style.borderColor = 'var(--green-dark)';
                });
                
                const wrapper = document.createElement('div');
                wrapper.appendChild(img);
                
                const label = document.createElement('div');
                label.textContent = item.type.toUpperCase();
                label.style.fontSize = '10px';
                label.style.textAlign = 'center';
                label.style.color = 'var(--muted)';
                label.style.fontWeight = 'bold';
                wrapper.appendChild(label);
                
                extractedGarments.appendChild(wrapper);
            });
            extractedGarments.classList.remove('hidden');
        })
        .catch(err => {
            alert(err.message);
        })
        .finally(() => {
            extractLoader.classList.add('hidden');
            extractGarmentBtn.disabled = false;
        });
    });

    fitButton.addEventListener('click', () => {
        const formData = new FormData();
        formData.append('file', personFile);
        if (garmentFile) {
            formData.append('clothing_file', garmentFile);
        } else if (garmentUrl) {
            formData.append('clothing_reference', garmentUrl);
        }
        formData.append('cloth_type', $('clothingTypeSelect').value);
        formData.append('ai_mode', 'tryon');
        
        fitButton.disabled = true;
        loader.classList.remove('hidden');
        message.textContent = '';
        
        fetch('/process', { method: 'POST', body: formData })
            .then(response => response.json().then(data => ({ ok: response.ok, data })))
            .then(({ ok, data }) => {
                if (!ok || !data.success) throw new Error(data.error || 'The fitting studio could not create this look.');
                addLook(data, true);
                message.textContent = 'Your look is ready.';
                document.location.hash = 'looks';
            })
            .catch(error => { message.textContent = error.message; })
            .finally(() => { loader.classList.add('hidden'); updateButton(); });
    });
    
    function openLightbox(index) {
        if (!galleryImages.length) return;
        lightboxIndex = (index + galleryImages.length) % galleryImages.length;
        $('lightboxImg').src = galleryImages[lightboxIndex];
        $('lightbox').classList.remove('hidden');
    }
    
    $('closeLightbox').addEventListener('click', () => $('lightbox').classList.add('hidden'));
    $('prevLightbox').addEventListener('click', () => openLightbox(lightboxIndex - 1));
    $('nextLightbox').addEventListener('click', () => openLightbox(lightboxIndex + 1));
    $('lightbox').addEventListener('click', event => {
        if (event.target.id === 'lightbox') $('lightbox').classList.add('hidden');
    });
    
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape') $('lightbox').classList.add('hidden');
        if (event.key === 'ArrowLeft') openLightbox(lightboxIndex - 1);
        if (event.key === 'ArrowRight') openLightbox(lightboxIndex + 1);
    });
    
    function addLook(data, prepend) {
        const empty = gallery.querySelector('.empty-state');
        if (empty) empty.remove();
        
        const card = document.createElement('article');
        card.className = 'gallery-card';
        
        const img = document.createElement('img');
        img.src = data.output_url;
        img.alt = 'Virtual try-on result';
        img.loading = 'lazy';
        img.addEventListener('click', () => openLightbox(galleryImages.indexOf(data.output_url)));
        
        const deleteButton = document.createElement('button');
        deleteButton.type = 'button';
        deleteButton.textContent = '×';
        deleteButton.title = 'Delete look';
        deleteButton.addEventListener('click', event => {
            event.stopPropagation();
            fetch(`/delete/${encodeURIComponent(data.output_filename)}`, { method: 'DELETE' }).then(response => {
                if (response.ok) {
                    card.remove();
                    galleryImages = galleryImages.filter(url => url !== data.output_url);
                }
            });
        });
        
        card.append(img, deleteButton);
        prepend ? gallery.prepend(card) : gallery.append(card);
        galleryImages = prepend ? [data.output_url, ...galleryImages] : [...galleryImages, data.output_url];
    }
    
    function addWardrobeItem(item, prepend) {
        const targetGalleryId = (item.type.includes('upper') || item.type === 'overall') ? 'wardrobeGalleryTops' : 'wardrobeGalleryBottoms';
        const targetGallery = $(targetGalleryId);
        
        const empty = targetGallery.querySelector('.empty-state');
        if (empty) empty.remove();
        
        const card = document.createElement('article');
        card.className = 'gallery-card';
        
        const img = document.createElement('img');
        img.src = item.url;
        img.alt = 'Extracted garment';
        img.loading = 'lazy';
        
        img.addEventListener('click', () => {
            garmentFile = null;
            garmentUrl = item.filename;
            $('clothingPreview').innerHTML = `<img src="${item.url}" alt="Extracted garment" style="object-fit: cover; width: 100%; height: 100%;"><span class="preview-label">${item.type}</span>`;
            $('clothingPreview').classList.add('has-image');
            $('clothingTypeSelect').value = item.type;
            updateButton();
            document.location.hash = 'studio';
        });
        
        const deleteButton = document.createElement('button');
        deleteButton.type = 'button';
        deleteButton.textContent = '×';
        deleteButton.title = 'Delete garment';
        deleteButton.addEventListener('click', event => {
            event.stopPropagation();
            fetch(`/delete_garment/${encodeURIComponent(item.filename)}`, { method: 'DELETE' }).then(response => {
                if (response.ok) {
                    card.remove();
                    if (targetGallery.children.length === 0) {
                        const defaultText = targetGalleryId === 'wardrobeGalleryTops' ? 'Extracted tops will appear here.' : 'Extracted bottoms will appear here.';
                        targetGallery.innerHTML = `<div class="empty-state">${defaultText}</div>`;
                    }
                }
            });
        });
        
        card.append(img, deleteButton);
        prepend ? targetGallery.prepend(card) : targetGallery.append(card);
    }
    
    fetch('/wardrobe')
        .then(response => response.json())
        .then(items => items.forEach(item => addWardrobeItem(item, false)))
        .catch(() => { /* non-critical */ });
        
    fetch('/history')
        .then(response => response.json())
        .then(items => items.forEach(item => addLook(item, false)))
        .catch(() => { message.textContent = 'Recent looks are unavailable right now.'; });
});
