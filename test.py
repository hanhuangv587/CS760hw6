torch.manual_seed(7777)

def generator_loss(output, true_label):
    ############ YOUR CODE HERE ##########
    loss_g = criterion(discriminator(fake), label_real(real.size(0)))
    return loss_g
    ######################################
    
def discriminator_loss(output, true_label):
    ############ YOUR CODE HERE ##########
    loss_d = criterion(discriminator(fake.detach()), label_fake(real.size(0))) + criterion(discriminator(real), label_real(real.size(0)))
    return loss_d
    ######################################
    

for epoch in range(epochs):
    loss_g = 0.0
    loss_d = 0.0
    for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size)):
        # get the data
        real, _ = data
        real = real.to(device)
    
        # train the discriminator
        for _ in range(k):
            # generate fake images
            fake = generator(create_noise(sample_size, nz))
            # get the loss for the discriminator
            loss_d = criterion(discriminator(fake.detach()), label_fake(sample_size)) + criterion(discriminator(real), label_real(real.size(0)))
            # optimize the discriminator
            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()
    
        # train the generator
        # generate fake images
        fake = generator(create_noise(real.size(0), nz))
        # get the loss for the generator
        loss_g = criterion(discriminator(fake), label_real(real.size(0)))
        # optimize the generator
        optim_g.zero_grad()
        loss_g.backward()
        optim_g.step()
    
    
    # create the final fake image for the epoch
    generated_img = generator(noise).cpu().detach()
    
    # make the images as grid
    generated_img = make_grid(generated_img)
    
    # visualize generated images
    if (epoch + 1) % 5 == 0:
        plt.imshow(generated_img.permute(1, 2, 0))
        plt.title(f'epoch {epoch+1}')
        plt.axis('off')
        plt.show()
    
    # save the generated torch tensor models to disk
    save_generator_image(generated_img, f"outputs/gen_img{epoch+1}.png")
    images.append(generated_img)
    epoch_loss_g = loss_g / bi # total generator loss for the epoch
    epoch_loss_d = loss_d / bi # total discriminator loss for the epoch
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)
    
    print(f"Epoch {epoch+1} of {epochs}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")